"""输出：静态地图（PNG）+ 自包含交互沙盘报告（HTML）。

两者都不依赖网络：HTML 里内嵌全部区块数据与绘图脚本，可以直接双击打开；
PNG 用 Pillow 绘制，适合贴进文档或作为缩略图。
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from . import materials, pft as pft_mod
from .verifiers import Foundation, Lander, Verdict, _foundation_at
from .world import World

BIOME_COLORS: dict[str, tuple[int, int, int]] = {
    "温带阔叶林": (46, 125, 50),
    "北方针叶林": (27, 94, 74),
    "硬叶灌丛": (140, 138, 54),
    "温带草原": (176, 168, 76),
    "C4 稀树草原": (198, 150, 62),
    "热带雨林": (20, 92, 48),
    "高山草甸": (137, 160, 120),
    "裸地/水体": (150, 148, 140),
}

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]


def _font(size: int):
    from PIL import ImageFont

    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# 地图栅格
# ---------------------------------------------------------------------------
def raster(world: World, key: str) -> np.ndarray:
    nx, ny = world.config.nx, world.config.ny
    out = np.zeros((ny, nx))
    for t in world.tiles:
        v = {
            "elev": t.elev,
            "wt": t.wt,
            "biomass": t.biomass,
            "npp": t.npp,
            "lai": t.lai,
            "discharge": t.discharge,
        }[key]
        out[t.iy, t.ix] = v
    return out


def fs_raster(world: World, foundation: Foundation | None = None) -> np.ndarray:
    f = foundation or Foundation()
    nx, ny = world.config.nx, world.config.ny
    out = np.zeros((ny, nx))
    for t in world.tiles:
        try:
            out[t.iy, t.ix] = _foundation_at(t, f, world)["fs"]
        except Exception:
            out[t.iy, t.ix] = 0.0
    return out


def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(a)), float(np.max(a))
    return (a - lo) / max(hi - lo, 1e-9)


def _terrain_color(z: np.ndarray) -> np.ndarray:
    """低地绿 → 坡地橄榄 → 高地灰褐 → 山脊灰白。"""
    stops = [
        (0.00, (44, 84, 52)),
        (0.30, (78, 112, 60)),
        (0.52, (128, 132, 78)),
        (0.72, (166, 152, 118)),
        (0.88, (198, 190, 170)),
        (1.00, (236, 236, 230)),
    ]
    h, w = z.shape
    out = np.zeros((h, w, 3), dtype=float)
    for i in range(h):
        for j in range(w):
            v = float(np.clip(z[i, j], 0, 1))
            for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
                if v <= t1:
                    k = (v - t0) / max(t1 - t0, 1e-9)
                    out[i, j] = [c0[0] + (c1[0] - c0[0]) * k,
                                 c0[1] + (c1[1] - c0[1]) * k,
                                 c0[2] + (c1[2] - c0[2]) * k]
                    break
            else:
                out[i, j] = stops[-1][1]
    return out.astype(np.uint8)


def _hillshade(z: np.ndarray, az: float = 315.0, alt: float = 45.0) -> np.ndarray:
    gy, gx = np.gradient(z)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy) * 6)
    aspect = np.arctan2(-gx, gy)
    az_r, alt_r = math.radians(az), math.radians(alt)
    shade = np.sin(alt_r) * np.sin(slope) + np.cos(alt_r) * np.cos(slope) * np.cos(
        az_r - aspect
    )
    return np.clip((shade + 1) / 2, 0, 1)


# ---------------------------------------------------------------------------
# PNG
# ---------------------------------------------------------------------------
def _biome_raster(world: World) -> np.ndarray:
    nx, ny = world.config.nx, world.config.ny
    out = np.zeros((ny, nx, 3), dtype=np.uint8)
    for t in world.tiles:
        out[t.iy, t.ix] = BIOME_COLORS.get(t.biome, (120, 120, 120))
    return out


def render_png(
    world: World,
    verdicts: list[Verdict],
    path: str | Path,
    *,
    before: World | None = None,
    foundation: Foundation | None = None,
    sites: list[tuple[float, float, str]] | None = None,
    scale: int = 9,
) -> Path:
    """3×2 面板：地形 / 地下水位 / 植被对照，以及替换前后的承载力与群区。"""
    from PIL import Image, ImageDraw

    nx, ny = world.config.nx, world.config.ny
    w_map = nx * scale
    base = before or world

    elev = raster(world, "elev")
    shade = _hillshade(elev)
    lo, hi = np.percentile(elev, [3, 97])
    elev_n = np.clip((elev - lo) / max(hi - lo, 1e-9), 0, 1)
    # 用归一化后的高程重新取色，避免整体过亮/过暗
    terrain = _terrain_color(elev_n)
    terrain = (terrain * (0.45 + 0.55 * shade[..., None])).astype(np.uint8)

    biome_before = _biome_raster(base)
    biome_after = _biome_raster(world)

    wt = _norm(raster(world, "wt"))
    water = np.stack([(60 + 40 * (1 - wt)), (110 + 70 * (1 - wt)),
                      (170 + 60 * (1 - wt))], axis=-1).astype(np.uint8)

    def fs_values(w: World) -> np.ndarray:
        return fs_raster(w, foundation)

    def fs_img(fs: np.ndarray) -> np.ndarray:
        fn = np.clip(fs / 4.0, 0, 1)
        return np.stack([
            np.clip(220 * (1 - fn) + 30, 0, 255),
            np.clip(60 + 160 * fn, 0, 255),
            np.clip(60 + 40 * fn, 0, 255),
        ], axis=-1).astype(np.uint8)

    def delta_img(d: np.ndarray) -> np.ndarray:
        """替换带来的承载力变化：蓝=变好，红=变差，灰=无变化。"""
        k = np.clip(d / 2.0, -1, 1)
        r = np.where(k < 0, 210, 200 - 150 * k)
        g = np.where(k < 0, 200 + 40 * k, 200 + 20 * k)
        b = np.where(k < 0, 60 - 30 * k, 210)
        return np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255),
                         np.clip(b, 0, 255)], axis=-1).astype(np.uint8)

    fs_before, fs_after = fs_values(base), fs_values(world)

    veg = np.clip(_norm(raster(world, "biomass")) * 1.2, 0, 1)
    veg_img = np.stack([70 * veg + 30, 60 + 150 * veg, 50 + 60 * veg],
                       axis=-1).astype(np.uint8)

    panels = [
        ("地形与地质", terrain),
        ("生物群区（替换前）", biome_before),
        ("生物群区（替换后）", biome_after),
        ("地下水位 / 汇流", water),
        ("承载力安全系数（替换后）", fs_img(fs_after)),
        ("承载力变化 Δ（蓝=变好，红=变差）", delta_img(fs_after - fs_before)),
    ]

    pad, header, footer = 16, 52, 52
    cols = 3
    total_w = pad + cols * (w_map + pad)
    total_h = header + 2 * (w_map + 26 + pad) + footer
    img = Image.new("RGB", (total_w, total_h), (20, 22, 26))
    draw = ImageDraw.Draw(img)
    f_title = _font(20)
    f_small = _font(13)
    draw.text((pad, 14), "BLOCKWORLD · 区块化地质—生态沙盘", font=f_title,
              fill=(235, 238, 242))

    for k, (title, arr) in enumerate(panels):
        col, row = k % cols, k // cols
        x0 = pad + col * (w_map + pad)
        y0 = header + row * (w_map + 26 + pad)
        tile = Image.fromarray(arr).resize((w_map, w_map), Image.NEAREST)
        img.paste(tile, (x0, y0))
        draw.rectangle([x0, y0, x0 + w_map, y0 + w_map], outline=(70, 74, 82))
        draw.text((x0 + 4, y0 + w_map + 4), title, font=f_small, fill=(200, 205, 212))
        # 在两张承载力图上标出验证场地
        if sites and "承载力" in title:
            kx = scale / world.config.cell
            for sx, sy, label in sites:
                px, py = x0 + sx * kx, y0 + sy * kx
                r = 13
                draw.ellipse([px - r, py - r, px + r, py + r], outline=(255, 255, 255), width=2)
                draw.text((px + r + 3, py - 7), label, font=f_small, fill=(255, 255, 255))

    y = total_h - footer + 6
    passed = sum(1 for v in verdicts if v.passed)
    draw.text((pad, y),
              f"判据：{passed}/{len(verdicts)} 通过　重力 {world.config.gravity:.2f} m/s²　"
              f"区块 {nx}×{ny} @ {world.config.cell:.0f} m　"
              f"钻孔 {world.config.n_boreholes} 个　"
              f"表层底界误差 {world.meta.get('topsoil_depth_mae_m', '—')} m",
              font=f_small, fill=(190, 196, 204))
    draw.text((pad, y + 20),
              "红色 = 基础承载力不足或沉降超限；绿色 = 有余量。替换后的判据变化即"
              "“是否值得换”的答案。", font=f_small, fill=(130, 138, 148))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BLOCKWORLD · 区块沙盘报告</title>
<style>
:root{--bg:#0f1115;--panel:#171a20;--line:#262b33;--ink:#e6e9ee;--dim:#98a0ac;
--ok:#3ddc84;--bad:#ff6b6b;--warn:#ffc857;--accent:#5aa9ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 "Segoe UI","Microsoft YaHei",sans-serif}
header{padding:20px 26px;border-bottom:1px solid var(--line);display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
h1{font-size:19px;margin:0;letter-spacing:.5px}
.sub{color:var(--dim);font-size:12.5px}
main{display:grid;grid-template-columns:minmax(360px,1fr) minmax(340px,460px);gap:18px;padding:18px 26px 40px}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-bottom:14px}
.card h2{font-size:13px;margin:0 0 10px;color:var(--dim);font-weight:600;letter-spacing:.4px;text-transform:uppercase}
canvas{width:100%;height:auto;border-radius:10px;background:#0b0d10;cursor:crosshair}
.layers{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
button{background:#1d222a;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:6px 12px;font-size:12.5px;cursor:pointer}
button.on{border-color:var(--accent);color:var(--accent)}
.verdict{display:flex;gap:10px;align-items:flex-start;padding:10px 0;border-bottom:1px dashed var(--line)}
.verdict:last-child{border-bottom:none}
.dot{width:9px;height:9px;border-radius:50%;margin-top:7px;flex:none}
.ok{background:var(--ok)}.bad{background:var(--bad)}
.vt{font-weight:600}.vd{color:var(--dim);font-size:12.5px}
.metrics{display:flex;gap:14px;flex-wrap:wrap;margin-top:6px;color:var(--dim);font-size:12px}
.metrics b{color:var(--ink);font-weight:600}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{text-align:left;padding:6px 4px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--dim);font-weight:500}
.bar{height:6px;background:#232830;border-radius:3px;overflow:hidden;min-width:70px}
.bar i{display:block;height:100%;background:var(--accent)}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--dim);margin-top:10px}
.legend span{display:inline-flex;gap:6px;align-items:center}
.sw{width:10px;height:10px;border-radius:2px;display:inline-block}
code{background:#1d222a;padding:1px 5px;border-radius:4px;font-size:12px}
.hint{color:var(--dim);font-size:12px;margin-top:8px}
</style></head><body>
<header>
  <h1>BLOCKWORLD · 区块化地质—生态沙盘</h1>
  <div class="sub" id="sub"></div>
</header>
<main>
  <section>
    <div class="card">
      <h2>区块视图</h2>
      <canvas id="map" width="720" height="720"></canvas>
      <div class="layers" id="layers"></div>
      <div class="legend" id="legend"></div>
      <div class="hint" id="probe">点击区块查看属性</div>
    </div>
    <div class="card">
      <h2>替换日志（事务式，可回滚）</h2>
      <table id="log"></table>
    </div>
    <div class="card">
      <h2>敏感性排序（Spearman ρ）</h2>
      <table id="sens"></table>
      <div class="hint" id="flips"></div>
    </div>
  </section>
  <section>
    <div class="card"><h2>目的验证</h2><div id="verdicts"></div></div>
    <div class="card"><h2>场景</h2><div id="scenario" class="vd"></div></div>
    <div class="card"><h2>生物推演</h2><div id="ecol" class="vd"></div></div>
  </section>
</main>
<script>
const D = __DATA__;
const BIOME = __BIOME__;
const nx = D.nx, ny = D.ny, N = nx*ny;
const cv = document.getElementById('map'), ctx = cv.getContext('2d');
const LAYERS = [['terrain','地形'],['biome','生物群区'],['water','地下水位'],['fs','承载力FS'],['biomass','植被'],['animals','动物']];
const fromHash = location.hash.replace('#','');
let layer = LAYERS.some(([k])=>k===fromHash) ? fromHash : 'terrain';

function ramp(v, stops){ // stops: [[t,[r,g,b]],...]
  v = Math.max(0, Math.min(1, v));
  for(let i=0;i<stops.length-1;i++){
    const [t0,c0]=stops[i], [t1,c1]=stops[i+1];
    if(v<=t1){ const k=(v-t0)/Math.max(t1-t0,1e-9);
      return [c0[0]+(c1[0]-c0[0])*k, c0[1]+(c1[1]-c0[1])*k, c0[2]+(c1[2]-c0[2])*k]; }
  }
  return stops[stops.length-1][1];
}
function norm(a){ let lo=Math.min(...a), hi=Math.max(...a); return a.map(v=>(v-lo)/Math.max(hi-lo,1e-9)); }
function normPct(a, p0=3, p1=97){
  const s=[...a].sort((x,y)=>x-y);
  const lo=s[Math.floor(s.length*p0/100)], hi=s[Math.floor(s.length*p1/100)];
  return a.map(v=>Math.max(0,Math.min(1,(v-lo)/Math.max(hi-lo,1e-9))));
}

const ELEV = normPct(D.elev), WT = norm(D.wt), FS = D.fs, BIO = D.biomass;
const TERRAIN = [[0,[52,96,58]],[.45,[92,132,66]],[.7,[150,146,96]],[1,[226,222,214]]];
const FSC = [[0,[214,64,64]],[.45,[224,168,60]],[.75,[150,200,90]],[1,[60,190,120]]];
const WTC = [[0,[46,86,140]],[.5,[84,150,196]],[1,[186,214,232]]];

function draw(){
  const cs = 720/nx;
  for(let iy=0;iy<ny;iy++) for(let ix=0;ix<nx;ix++){
    const j = iy*nx+ix; let c;
    if(layer==='terrain'){ c = ramp(ELEV[j], TERRAIN); }
    else if(layer==='biome'){ c = BIOME[D.biome[j]] || [120,120,120]; }
    else if(layer==='water'){ c = ramp(1-WT[j], WTC); }
    else if(layer==='fs'){ c = ramp(Math.min(FS[j]/4,1), FSC); }
    else { const b = BIO[j]; c = b>0.15 ? [40+b*14, 90+b*10, 52] : [58,58,62]; }
    ctx.fillStyle = `rgb(${c[0]|0},${c[1]|0},${c[2]|0})`;
    ctx.fillRect(ix*cs, iy*cs, Math.ceil(cs), Math.ceil(cs));
    if(D.edited[j]){ ctx.strokeStyle='rgba(255,255,255,.18)'; ctx.lineWidth=1;
      ctx.strokeRect(ix*cs+0.5, iy*cs+0.5, cs-1, cs-1); }
  }
  if(layer==='animals'){
    for(const [ix,iy,n] of D.animalMarkers){
      ctx.fillStyle='rgba(255,220,120,.9)';
      ctx.beginPath(); ctx.arc((ix+.5)*cs,(iy+.5)*cs, Math.min(cs*.5, 2+n*.35), 0, 7); ctx.fill();
    }
  }
}

const lb = document.getElementById('layers');
LAYERS.forEach(([k,label])=>{ const b=document.createElement('button'); b.textContent=label;
  if(k===layer) b.className='on';
  b.onclick=()=>{ layer=k; [...lb.children].forEach(x=>x.className=''); b.className='on'; draw(); legend(); };
  lb.appendChild(b); });
function legend(){
  const el=document.getElementById('legend'); let html='';
  if(layer==='biome'){ for(const [k,v] of Object.entries(BIOME)) html+=`<span><i class="sw" style="background:rgb(${v})"></i>${k}</span>`; }
  else if(layer==='fs'){ html='<span>红=FS&lt;1.5 失稳风险</span><span>黄=1.5–2.5</span><span>绿=FS&gt;3 稳定</span>'; }
  else if(layer==='water'){ html='<span>深蓝=水位浅（0–1 m）</span><span>浅蓝=深（&gt;8 m）</span>'; }
  else if(layer==='terrain'){ html='<span>绿=低地/河谷</span><span>褐=坡地</span><span>灰白=山脊</span>'; }
  else if(layer==='animals'){ html='<span>圆点=动物个体，大小∝数量</span>'; }
  else { html='<span>亮度∝生物量（kg C/m²）</span>'; }
  el.innerHTML=html;
}

cv.onclick = e => {
  const r=cv.getBoundingClientRect(), cs=720/nx;
  const ix=Math.floor((e.clientX-r.left)/r.width*nx), iy=Math.floor((e.clientY-r.top)/r.height*ny);
  const j=iy*nx+ix;
  document.getElementById('probe').textContent =
    `区块(${ix},${iy})　高程 ${D.elev[j].toFixed(1)} m　坡度 ${D.slope[j].toFixed(1)}°　`+
    `生物群区 ${D.biomeName[j]}　生物量 ${D.biomass[j].toFixed(2)} kgC/m²　`+
    `水位 ${D.wt[j].toFixed(2)} m　承载力FS ${D.fs[j].toFixed(2)}　`+
    `0–2 m 地层 ${D.topmat[j]}`;
};

document.getElementById('sub').textContent =
  `${D.meta.cells} 区块 · ${D.nx}×${D.ny} @ ${D.meta.cell_m} m · 重力 ${D.meta.gravity} m/s² · 钻孔 ${D.meta.boreholes} · 数据 ${D.meta.source} · 表层底界误差 ${D.meta.topsoil_mae} m`;

// 判据卡片
function fmt(v){
  if(typeof v === 'number') return Number.isInteger(v) ? v : v.toFixed(2);
  if(v && typeof v === 'object') return Object.entries(v).map(([k,x])=>`${k}:${x}`).join(' / ');
  return String(v);
}
const V = D.verdicts;
document.getElementById('verdicts').innerHTML = V.map(v=>`
  <div class="verdict"><div class="dot ${v.passed?'ok':'bad'}"></div><div>
    <div class="vt">${v.title} <span class="vd">${v.passed?'通过':'未通过'}</span></div>
    <div class="vd">${v.reasons.join('；')}</div>
    <div class="metrics">${Object.entries(v.metrics).slice(0,7).map(([k,x])=>
      `<span>${k}=<b>${fmt(x)}</b></span>`).join('')}</div>
  </div></div>`).join('');

document.getElementById('scenario').innerHTML = D.scenario;

// 替换日志
const L = D.log;
document.getElementById('log').innerHTML = L.length ? `<tr><th>#</th><th>操作</th><th>区域</th><th>详情</th><th>影响区块</th></tr>` +
  L.map(r=>`<tr><td>${r.index}</td><td>${r.op}</td><td>${r.region}</td>
  <td>${Object.entries(r.detail).map(([k,v])=>`${k}=${v}`).join('，')}</td>
  <td>${r.tiles_affected}</td></tr>`).join('') : `<tr><td class="vd">本次运行没有替换操作</td></tr>`;

// 敏感性
const S = D.sensitivity || [];
document.getElementById('sens').innerHTML = S.length ? `<tr><th>参数</th><th>ρ</th><th></th><th>翻转</th></tr>` +
  S.map(r=>`<tr><td>${r.label}</td><td>${r.rho.toFixed(2)}</td>
  <td><div class="bar"><i style="width:${Math.min(100,Math.abs(r.rho)*100)}%"></i></div></td>
  <td>${r.flips}</td></tr>`).join('') : `<tr><td class="vd">未运行敏感性扫描</td></tr>`;
const fl = (D.flipText||[]);
document.getElementById('flips').innerHTML = fl.length ? fl.join('<br>') : '';

document.getElementById('ecol').innerHTML = D.ecology;

draw(); legend();
</script>
</body></html>
"""


def render_html(
    world: World,
    verdicts: list[Verdict],
    path: str | Path,
    *,
    log: list[dict] | None = None,
    sensitivity: list[dict] | None = None,
    flip_text: list[str] | None = None,
    scenario: str = "",
    ecology: str = "",
    foundation: Foundation | None = None,
    scale: int = 720,
) -> Path:
    nx, ny = world.config.nx, world.config.ny
    elev = raster(world, "elev").ravel()
    wt = raster(world, "wt").ravel()
    biomass = raster(world, "biomass").ravel()
    fs = fs_raster(world, foundation).ravel()
    slope = np.array([world.slope_deg(t) for t in world.tiles])
    topmat = [t.col.layers[0][1].name for t in world.tiles]
    biome_name = [t.biome for t in world.tiles]

    animal_markers = [[t.ix, t.iy, len(t.animals)] for t in world.tiles if t.animals]
    data = {
        "nx": nx, "ny": ny,
        "elev": [round(float(v), 2) for v in elev],
        "wt": [round(float(v), 2) for v in wt],
        "biomass": [round(float(v), 3) for v in biomass],
        "fs": [round(float(v), 3) for v in fs],
        "slope": [round(float(v), 2) for v in slope],
        "biome": biome_name,
        "topmat": topmat,
        "edited": [bool({"set_material", "replace_component", "set_elevation",
                         "set_water_table"} & set(t.edit_tags)) for t in world.tiles],
        "animalMarkers": animal_markers,
        "meta": {
            "cells": len(world.tiles),
            "cell_m": world.config.cell,
            "gravity": round(world.config.gravity, 2),
            "boreholes": world.config.n_boreholes,
            "dem_rmse": world.meta.get("dem_rmse_m", "—"),
            "topsoil_mae": world.meta.get("topsoil_depth_mae_m", "—"),
            "source": world.meta.get("source", "user-data"),
        },
        "verdicts": [v.to_dict() for v in verdicts],
        "log": log or [],
        "sensitivity": sensitivity or [],
        "flipText": flip_text or [],
        "scenario": scenario.replace("\n", "<br>"),
        "ecology": ecology.replace("\n", "<br>"),
    }
    html = (_HTML
            .replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__BIOME__", json.dumps({k: list(v) for k, v in BIOME_COLORS.items()},
                                             ensure_ascii=False)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
