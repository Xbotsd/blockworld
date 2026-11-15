"""端到端演示：建世界 → 推演 → 目的验证 → 替换 → 判据翻转 → 敏感性 → 报告。"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from . import animals, edits, hydrology, materials, pft as pft_mod, report, vegetation
from . import sensitivity as S
from . import verifiers as V
from .world import Region, World, WorldConfig, build

LANDER_BASE = dict(mass_kg=3000.0, n_pads=3, touchdown_v=2.5,
                   leg_stroke=0.35, fs_required=2.0)
FOUNDATION_BASE = dict(B=1.5, L=1.5, D=1.5, load_kN=1200.0,
                       fs_required=3.0, settle_allow_mm=25.0)


def _pick(world: World, highest: bool = False) -> tuple[float, float]:
    t = (max if highest else min)(world.tiles, key=lambda x: x.elev)
    return t.x, t.y


def _pick_flat(world: World, max_elev: float = 120.0, radius: int = 1):
    """着陆场选址：低地平原里最平坦的一处（1 邻域内最大坡度最小）。"""
    best: tuple[float, object] | None = None
    for t in world.tiles:
        if t.elev > max_elev:
            continue
        nb = [t]
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                jx, jy = t.ix + dx, t.iy + dy
                if 0 <= jx < world.config.nx and 0 <= jy < world.config.ny:
                    nb.append(world.tile(jx, jy))
        m = max(world.slope_deg(x) for x in nb)
        if best is None or m < best[0]:
            best = (m, t)
    return best[1] if best else world.tiles[0]


def _pick_softest(world: World) -> tuple[float, float]:
    """按真实选址逻辑挑场地：缓坡低地（河谷/平地）里最软的地基。"""
    cands = [t for t in world.tiles
             if world.slope_deg(t) < 8.0 and t.elev < 60.0]
    if not cands:
        cands = world.tiles
    t = min(cands, key=lambda x: (x.col.material_at(1.5).E,
                                  x.col.material_at(1.5).c))
    return t.x, t.y


def _fmt(v: V.Verdict) -> str:
    return f"{'通过' if v.passed else '未通过'}（{v.reasons[0] if v.reasons else ''}）"


def _set_surface(world: World, region: Region, prop: str, value: float) -> None:
    """改区域内表层材料的单个参数（敏感性扫描用）。"""
    for t in world.region_tiles(region):
        th, m = t.col.layers[0]
        m2 = m.with_params(**{prop: value})
        errs, _ = m2.check()
        if errs:
            raise materials.DomainError("；".join(errs))
        t.col.layers[0] = (th, m2)


def _landing(world: World, region: Region, pad: float) -> V.Verdict:
    return V.verify_landing(world, region, V.Lander(pad_diameter=pad, **LANDER_BASE))


def run(out_dir: str | Path = "out", *, spin_years: int = 55,
        animal_years: int = 6, seed: int | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = WorldConfig()
    if seed is not None:
        cfg.seed = seed
    world = build(cfg)

    # --- 1) 生态-水文耦合推演 -----------------------------------------
    hydrology.step(world)
    traj = vegetation.spin_up(world, years=spin_years)
    ecol_before = animals.run(world, years=animal_years, seed=cfg.seed + 1, initial=200)

    valley = Region.circle(*_pick_softest(world), 200.0)
    flat = _pick_flat(world)
    land_region = Region.circle(flat.x, flat.y, 150.0)

    foundation = V.Foundation(**FOUNDATION_BASE)
    lander = V.Lander(pad_diameter=0.25, **LANDER_BASE)

    baseline = V.evaluate_all(world, site=valley, foundation=foundation,
                              lander=lander, animal_stats=ecol_before)
    world_before = world.snapshot()

    # --- 2) 场景 A：基础设施（软土 → 换填）---------------------------
    infra_before = V.verify_infrastructure(world, valley, foundation)
    session = edits.EditSession(world)
    session.set_material(valley, "gravel_dense", top_depth=0.0, bottom_depth=8.0)
    infra_after = V.verify_infrastructure(world, valley, foundation)

    # --- 3) 场景 B：着陆（月球重力 + 风化层 → 胶结处理）---------------
    session.set_gravity(1.62)
    session.set_material(land_region, "lunar_regolith", top_depth=0.0, bottom_depth=2.5)
    landing_before = V.verify_landing(world, land_region, lander)
    session.set_material(land_region, "cemented_regolith", top_depth=0.0, bottom_depth=2.5)
    landing_after = V.verify_landing(world, land_region, lander)
    landing_pad = _landing(world, land_region, 0.9)

    # --- 4) 场景 C：生物群落替换（仅南部平原）-------------------------
    south = Region.rect(0.0, 0.0, world.config.width, world.config.height * 0.55)
    biomes_before = vegetation.biome_counts(world)
    session.set_precip(south, factor=0.55)
    session.set_pft(south, "c4_grass", mode="replace")
    hydrology.step(world)
    vegetation.spin_up(world, years=30)
    biomes_after = vegetation.biome_counts(world)
    ecol_after = animals.run(world, years=4, seed=cfg.seed + 2, initial=160)

    # 恢复地球重力后再评估基础设施（替换是可撤销的工程变量）
    session.set_gravity(9.81)

    # --- 5) 敏感性 ----------------------------------------------------
    land_sweeps = [
        S.oat_simple(world, "pad_diameter", "足垫直径 (m)",
                     [0.15, 0.2, 0.25, 0.35, 0.5, 0.7, 0.9],
                     lambda w, v: w.meta.__setitem__("_sweep_pad", v),
                     lambda w: _landing(w, land_region, _pad_of(w))),
        S.oat_simple(world, "surface_c", "表层黏聚力 c (kPa)",
                     [0.2, 1.0, 5.0, 12.0, 25.0, 45.0],
                     lambda w, v: _set_surface(w, land_region, "c", v),
                     lambda w: _landing(w, land_region, 0.25)),
        S.oat_simple(world, "surface_phi", "表层内摩擦角 φ (°)",
                     [28, 31, 34, 37, 40],
                     lambda w, v: _set_surface(w, land_region, "phi", v),
                     lambda w: _landing(w, land_region, 0.25)),
        S.oat_simple(world, "gravity", "局部重力 (m/s²)",
                     [0.5, 1.0, 1.62, 2.5, 3.71, 9.81],
                     lambda w, v: S.set_gravity(w, v),
                     lambda w: _landing(w, land_region, 0.25)),
    ]
    infra_sweeps = [
        S.oat_simple(world, "fill_c", "换填层黏聚力 c (kPa)",
                     [0.0, 5.0, 15.0, 30.0, 50.0, 80.0],
                     lambda w, v: _set_surface(w, valley, "c", v),
                     lambda w: V.verify_infrastructure(w, valley, foundation)),
        S.oat_simple(world, "fill_phi", "换填层内摩擦角 φ (°)",
                     [26, 30, 33, 36, 40, 43],
                     lambda w, v: _set_surface(w, valley, "phi", v),
                     lambda w: V.verify_infrastructure(w, valley, foundation)),
    ]
    b_rows = []
    for b in [1.0, 1.5, 2.0, 2.5, 3.0]:
        v = V.verify_infrastructure(world, valley,
                                    V.Foundation(B=b, L=b, D=1.5, load_kN=1200.0))
        b_rows.append(S.SweepRow(float(b), float(v.metrics["worst_fs"]),
                                 v.passed, v.margin, list(v.reasons)))
    infra_sweeps.append(S.SweepResult("footing_B", "基础宽度 B (m)", b_rows))

    all_sweeps = land_sweeps + infra_sweeps
    ranking = S.rank(all_sweeps)

    grid = S.grid_2d(
        world, [0.2, 1, 3, 8, 15, 25, 40], [28, 31, 34, 37, 40],
        lambda w, c, phi: (_set_surface(w, land_region, "c", c),
                           _set_surface(w, land_region, "phi", phi)),
        lambda w: _landing(w, land_region, 0.25),
    )

    # --- 6) 汇总 ------------------------------------------------------
    eco_v = V.verify_ecology(world, Region.whole(), animal_stats=ecol_after)
    verdicts = [
        V.verify_infrastructure(world, valley, foundation),
        V.verify_slope(world, Region.whole()),
        V.verify_landing(world, land_region, lander),
        eco_v,
    ]

    scenario_text = "\n".join([
        f"世界：{world.config.nx}×{world.config.ny} 区块 @ {world.config.cell:.0f} m，"
        f"重力 {world.config.gravity:.2f} m/s²，钻孔 {world.config.n_boreholes} 个",
        f"A 基础设施（河谷软土，1.5×1.5 m 基础，1200 kN）：替换前 {_fmt(infra_before)}",
        f"　→ 0–8 m 换填密实砂砾：{_fmt(infra_after)}",
        f"B 着陆（月球重力，3000 kg / 3 足垫 / d=0.25 m）：替换前 {_fmt(landing_before)}",
        f"　→ 2.5 m 胶结处理：{_fmt(landing_after)}",
        f"　→ 仅把足垫加大到 0.90 m：{_fmt(landing_pad)}",
        f"C 生物群落：降水 ×0.55 + C4 草地替换，{len(biomes_before)} 类群区 → "
        f"{len(biomes_after)} 类",
    ])

    ecology_text = "\n".join([
        f"平均 NPP {eco_v.metrics['mean_npp']:.2f} kgC/m²/yr，"
        f"平均生物量 {eco_v.metrics['mean_biomass']:.2f} kgC/m²，"
        f"Shannon {eco_v.metrics['shannon']:.2f}",
        f"动物：{ecol_after['population']} 只，"
        f"最大存活体重 {ecol_after['largest_survivor_kg']:.2f} kg，"
        f"占据 {ecol_after['occupancy']*100:.0f}% 区块"
        f"（替换前最大存活体重 {ecol_before['largest_survivor_kg']:.2f} kg）",
    ])

    flip_text = [
        f"{r.label}: {f['from']} → {f['to']} 之间发生{f['direction']}，临界值 ≈ {f['critical']}"
        for r in all_sweeps for f in r.flips
    ]
    sens_payload = sorted(
        ({"factor": r.factor, "label": r.label, "rho": round(r.rho, 3),
          "flips": len(r.flips)} for r in all_sweeps),
        key=lambda d: -abs(d["rho"]),
    )

    summary = {
        "world": world.summary(),
        "baseline": [v.to_dict() for v in baseline],
        "scenario_A_infrastructure": {"before": infra_before.to_dict(),
                                      "after": infra_after.to_dict()},
        "scenario_B_landing": {"before": landing_before.to_dict(),
                               "after": landing_after.to_dict(),
                               "pad_enlarged": landing_pad.to_dict()},
        "scenario_C_ecology": {
            "biomes_before": biomes_before,
            "biomes_after": biomes_after,
            "animals_before": {k: v for k, v in ecol_before.items() if k != "trajectory"},
            "animals_after": {k: v for k, v in ecol_after.items() if k != "trajectory"},
            "verdict": eco_v.to_dict(),
        },
        "verdicts": [v.to_dict() for v in verdicts],
        "sensitivity": sens_payload,
        "sweeps": [r.to_dict() for r in all_sweeps],
        "grid_2d": grid,
        "vegetation_trajectory_tail": traj["trajectory"][-3:],
        "slope": V.verify_slope(world, Region.whole()).to_dict(),
        "materials_table": materials.table(),
        "pft_table": pft_mod.table(),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out / "sensitivity.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["factor", "label", "value", "worst_fs", "passed", "margin"])
        for r in all_sweeps:
            for row in r.rows:
                w.writerow([r.factor, r.label, row.value, round(row.metric, 4),
                            row.passed, round(row.margin, 4)])

    report.render_png(world, verdicts, out / "sandbox.png",
                      before=world_before, foundation=foundation,
                      sites=[(valley.cx, valley.cy, "A 基础设施"),
                             (land_region.cx, land_region.cy, "B 着陆")])
    report.render_html(
        world, verdicts, out / "report.html",
        log=session.log(), sensitivity=sens_payload, flip_text=flip_text,
        scenario=scenario_text, ecology=ecology_text, foundation=foundation,
    )
    return summary


def _pad_of(world: World) -> float:
    return float(world.meta.get("_sweep_pad", 0.25))


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser("blockworld", description="区块化地质—生态沙盘")
    ap.add_argument("--out", default="out")
    ap.add_argument("--years", type=int, default=55)
    ap.add_argument("--animals", type=int, default=6)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    s = run(args.out, spin_years=args.years, animal_years=args.animals, seed=args.seed)
    print(json.dumps({
        "verdicts": {v["title"]: v["passed"] for v in s["verdicts"]},
        "top_factors": s["sensitivity"][:3],
        "outputs": [str(Path(args.out) / n)
                    for n in ("report.html", "sandbox.png", "summary.json")],
    }, ensure_ascii=False, indent=2))
    return 0
