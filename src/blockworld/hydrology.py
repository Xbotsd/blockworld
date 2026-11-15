"""水文：降水 → 蒸散/入渗/径流 → 汇流 → 地下水位。

刻意保持"可解释的简化"：PET 用 Hargreaves，蒸散分配用 Budyko 框架，产流用
坡度+渗透性修正的径流系数，汇流用 D8 顺序累积。所有公式都是公开形式，参数
可替换，便于检验"替换后判据是否翻转"。
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from . import pft as pft_mod
from .world import Tile, World


def extraterrestrial_radiation_mm(lat_deg: float, doy: int) -> float:
    """日地外辐射，折算为等效蒸发水层（mm/day）。FAO-56 公式。"""
    phi = math.radians(lat_deg)
    delta = 0.409 * math.sin(2 * math.pi * doy / 365 - 1.39)
    dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365)
    x = -math.tan(phi) * math.tan(delta)
    ws = math.acos(min(1.0, max(-1.0, x)))
    ra_mj = (24 * 60 / math.pi) * 0.0820 * dr * (
        ws * math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.sin(ws)
    )
    return max(0.0, ra_mj * 0.408)


@lru_cache(maxsize=8192)
def annual_pet(temp_c: float, lat_deg: float = 35.0, amplitude: float = 9.0) -> float:
    """年潜在蒸散 mm（Hargreaves，逐月累加）。amplitude 为气温年较差半幅。"""
    total = 0.0
    for m in range(12):
        doy = int(15 + 30.4 * m)
        t = temp_c + amplitude * math.sin(2 * math.pi * (m - 1.5) / 12)
        # 日较差取 11 °C，随季节略变
        dtr = 9.0 + 3.0 * math.sin(2 * math.pi * (m - 1.5) / 12)
        total += 0.0023 * (t + 17.8) * math.sqrt(max(dtr, 1.0)) * extraterrestrial_radiation_mm(lat_deg, doy) * 30.4
    return max(total, 0.0)


def _budyko(precip: float, pet: float) -> float:
    """Budyko 曲线：实际蒸散 AET = f(P, PET) * P。"""
    if precip <= 0:
        return 0.0
    if pet <= 0:
        return 0.0
    r = pet / precip
    return precip * math.sqrt(max(0.0, r * math.tanh(1 / r) * (1 - math.exp(-r))))


def kc_for(t: Tile) -> float:
    """区块的作物系数：由植被构成决定（裸地→土面蒸发）。"""
    if not t.cohort:
        return 0.35
    kc = 0.0
    total = sum(c.biomass for c in t.cohort) or 1.0
    for c in t.cohort:
        p = pft_mod.get(c.pft)
        kc += (c.biomass / total) * (1.15 if p.is_woody else 1.05)
    return max(0.35, kc)


def step(world: World, years: float = 1.0) -> dict:
    """推进水文一步（默认 1 年），并把结果写回区块。

    * 蒸散：Budyko 框架，PET 由 Hargreaves 逐月累加；
    * 产流：降水减去实际蒸散，按表层渗透性与坡度分配入渗/地表径流；
    * 汇流：D8 顺序累积，得到汇流面积与流量；
    * 地下水位：由地形湿度指数 TWI = ln(a / tanβ) 决定（河谷浅、山脊深），
      再叠加当年入渗的少量抬升——这是"地形控制潜水面"的常用简化。
    """
    cfg = world.config
    n = len(world.tiles)
    cell = cfg.cell
    pet_cache: dict[float, float] = {}
    runoff_vol = np.zeros(n)

    for t in world.tiles:
        key = round(t.temp, 2)
        if key not in pet_cache:
            pet_cache[key] = annual_pet(t.temp)
        pet = pet_cache[key]
        p = t.precip
        kc = kc_for(t)
        pet_veg = pet * kc

        top = t.col.layers[0][1]
        slope = max(world.slope_deg(t), 0.4)
        infil_cap = min(1.0, 0.45 + 0.6 * math.log10(max(top.k, 1e-10) / 1e-5))
        infil_cap *= max(0.15, 1.0 - slope / 60.0)

        aet = _budyko(p, pet_veg)
        runoff = max(0.0, p - aet)
        infil = runoff * infil_cap
        surface = runoff - infil

        t.et = aet
        t.infil = infil
        t.runoff = surface
        runoff_vol[world.idx(t.ix, t.iy)] = surface / 1000.0 * cell**2 / (365 * 86400)

    # D8 汇流：按高程降序把水量与汇水面积累加到最陡下坡邻居
    order = sorted(world.tiles, key=lambda t: -t.elev)
    acc = np.ones(n)
    q = runoff_vol.copy()
    for t in order:
        i = world.idx(t.ix, t.iy)
        nb = world.neighbours(t.ix, t.iy)
        if not nb:
            continue
        low = min(nb, key=lambda x: x.elev)
        if low.elev < t.elev - 1e-6:
            j = world.idx(low.ix, low.iy)
            q[j] += q[i]
            acc[j] += acc[i]

    twi = np.empty(n)
    for t in world.tiles:
        i = world.idx(t.ix, t.iy)
        beta = math.radians(max(world.slope_deg(t), 0.6))
        a = max(acc[i], 1.0) * cell          # 单位等高线长度的汇水面积
        twi[i] = math.log(a / max(math.tan(beta), 1e-4))
    lo, hi = np.percentile(twi, [5, 95])
    span = max(hi - lo, 1e-6)
    for t in world.tiles:
        i = world.idx(t.ix, t.iy)
        wet = float(np.clip((twi[i] - lo) / span, 0.0, 1.0))
        recharge_bonus = min(1.5, t.infil / 400.0)        # 当年入渗抬升水位
        t.wt = float(np.clip(9.5 - 8.6 * wet - recharge_bonus, 0.4, 12.0))
        t.discharge = float(q[i])

    return {
        "mean_et_mm": float(np.mean([t.et for t in world.tiles])),
        "mean_runoff_mm": float(np.mean([t.runoff for t in world.tiles])),
        "mean_infil_mm": float(np.mean([t.infil for t in world.tiles])),
        "mean_wt_m": float(np.mean([t.wt for t in world.tiles])),
        "peak_discharge_ls": float(max(t.discharge for t in world.tiles) * 1000),
    }
