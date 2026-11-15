"""目的验证器：把"可行性"变成可计算的判据。

三个目的各自挂一组判据：
  * 基础设施：地基承载力（Terzaghi/Vesic 承载力公式）+ 弹性沉降 + 边坡稳定；
  * 着陆：足垫接触压力、动力放大后的承载安全系数、穿透深度、峰值减速度；
  * 生态：植被可持续性、生物群区多样性、营养级容量（最大存活体重）。

所有输出都带"裕度"（margin），便于做判据翻转检测与敏感性排序。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import materials, solvers
from .world import Region, Tile, World

GAMMA_W = 9.81   # 水的重度 kN/m³
G_EARTH = 9.81


# ---------------------------------------------------------------------------
# 结果对象
# ---------------------------------------------------------------------------
@dataclass
class Verdict:
    key: str
    title: str
    passed: bool
    margin: float                 # 归一化裕度，>0 表示通过且有余量
    metrics: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    inputs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # numpy 布尔/标量会让下游 JSON 序列化失败，这里统一成原生类型
        from .util import native

        self.passed = bool(self.passed)
        self.margin = float(self.margin)
        self.metrics = {k: native(v) for k, v in self.metrics.items()}
        self.inputs = {k: native(v) for k, v in self.inputs.items()}

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "passed": self.passed,
            "margin": round(self.margin, 4),
            "metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in self.metrics.items()},
            "reasons": self.reasons,
            "inputs": self.inputs,
        }


# ---------------------------------------------------------------------------
# 承载力系数
# ---------------------------------------------------------------------------
def bearing_factors(phi_deg: float) -> dict:
    """Vesic 承载力系数。φ=0 时退化为不排水解。"""
    phi = math.radians(max(0.0, min(phi_deg, 44.9)))
    if phi < 1e-6:
        return {"Nq": 1.0, "Nc": 5.14, "Ngamma": 0.0}
    nq = math.exp(math.pi * math.tan(phi)) * math.tan(math.pi / 4 + phi / 2) ** 2
    nc = (nq - 1.0) / math.tan(phi)
    ng = 2.0 * (nq + 1.0) * math.tan(phi)
    return {"Nq": nq, "Nc": nc, "Ngamma": ng}


def shape_factors(shape: str, phi_deg: float) -> tuple[float, float, float]:
    """Terzaghi 形状系数 (sc, sq, sγ)。"""
    if shape == "strip":
        return 1.0, 1.0, 1.0
    if shape == "circle":
        return 1.3, 1.2, 0.6
    return 1.3, 1.2, 0.8          # square / rectangular


def bearing_capacity(
    c: float, phi_deg: float, gamma: float, B: float, D: float = 0.0,
    shape: str = "square", L: float | None = None,
) -> dict:
    """单位面积极限承载力 qu (kPa)。

    qu = c·Nc·sc·dc + q·Nq·sq·dq + 0.5·γ·B·Nγ·sγ·dγ
    """
    L = L or B
    f = bearing_factors(phi_deg)
    sc, sq, sg = shape_factors(shape, phi_deg)
    if shape == "rect":
        sc, sq = 1.3, 1.2
        sg = 1 - 0.4 * B / max(L, 1e-9)
    dc = 1 + 0.35 * D / max(B, 1e-9)
    dq = dg = 1 + 0.25 * D / max(B, 1e-9)
    q = gamma * D
    term_c = c * f["Nc"] * sc * dc
    term_q = q * f["Nq"] * sq * dq
    term_g = 0.5 * gamma * B * f["Ngamma"] * sg * dg
    return {
        "qu_kpa": term_c + term_q + term_g,
        "term_cohesion": term_c,
        "term_surcharge": term_q,
        "term_selfweight": term_g,
        "factors": f,
    }


def elastic_settlement(q_kpa: float, B: float, E_mpa: float, nu: float,
                       shape: str = "square") -> float:
    """弹性瞬时沉降（mm）。I 取刚性/柔性之间的常用值。"""
    if E_mpa <= 0:
        return float("inf")
    I = 0.82 if shape == "circle" else 0.95
    return q_kpa * B * (1 - nu**2) * I / E_mpa


def soil_profile(col, top_depth: float, bottom_depth: float, wt: float | None) -> dict:
    """区间加权地层参数（水下用浮重度）。"""
    return col.weighted(top_depth, bottom_depth, water_table=wt)


# ---------------------------------------------------------------------------
# 目的 A：基础设施
# ---------------------------------------------------------------------------
@dataclass
class Foundation:
    B: float = 2.0            # 基础宽度 m
    L: float = 2.0
    D: float = 1.5            # 埋深 m
    load_kN: float = 1200.0   # 竖向荷载 kN
    shape: str = "square"     # square | strip | circle | rect
    fs_required: float = 3.0
    settle_allow_mm: float = 25.0


def _foundation_at(t: Tile, f: Foundation, world: World) -> dict:
    gamma_scale = world.config.gravity / G_EARTH
    prof = soil_profile(t.col, f.D, f.D + f.B, t.wt)
    c, phi, gamma = prof["c"], prof["phi"], prof["gamma"] * gamma_scale

    qu_fn = solvers.property_solver(
        "bearing_qu",
        lambda p: bearing_capacity(p["c"], p["phi"], p["gamma"], p["B"],
                                   p["D"], p["shape"], p["L"]),
    )
    bc = solvers.normalize_capacity(qu_fn({
        "c": c, "phi": phi, "gamma": gamma, "B": f.B, "D": f.D,
        "shape": f.shape, "L": f.L, "gravity": world.config.gravity,
    }))
    area = f.B * f.L
    q_app = f.load_kN / area + gamma * f.D
    fs = bc["qu_kpa"] / max(q_app, 1e-6)

    settle_fn = solvers.property_solver(
        "settlement_mm",
        lambda p: elastic_settlement(p["q"], p["B"], p["E"], p["nu"], p["shape"]),
    )
    s = float(settle_fn({"q": q_app, "B": f.B, "E": prof["E"],
                         "nu": prof["nu"], "shape": f.shape}))
    terms = bc.get("terms") or {
        k: bc[k] for k in ("term_cohesion", "term_surcharge", "term_selfweight")
        if k in bc
    }
    return {
        "tile": (t.ix, t.iy),
        "c": c, "phi": phi, "gamma": gamma, "E": prof["E"], "nu": prof["nu"],
        "qu_kpa": bc["qu_kpa"], "q_applied_kpa": q_app, "fs": fs,
        "settlement_mm": s,
        "terms": terms,
    }


def verify_infrastructure(
    world: World, region: Region, foundation: Foundation | None = None,
    *, n_sites: int = 12, slope_limit: float = 20.0,
) -> Verdict:
    """在区域内抽样布置基础，取最不利者作为判据。"""
    f = foundation or Foundation()
    tiles = world.region_tiles(region)
    if not tiles:
        raise ValueError("区域内没有区块")
    # 选址抽样：既覆盖全区，也强制包含最软的几处（对应"最不利点"语义）
    soft = sorted(tiles, key=lambda t: t.col.weighted(f.D, f.D + f.B, t.wt)["E"])[:6]
    step = max(1, len(tiles) // n_sites)
    merged = {t.ix * 10000 + t.iy: t for t in soft + tiles[::step][:n_sites]}
    sites = list(merged.values())

    rows = [_foundation_at(t, f, world) for t in sites]
    worst = min(rows, key=lambda r: r["fs"])
    slopes = [world.slope_deg(t) for t in sites]
    max_slope = max(slopes)

    fs_ok = worst["fs"] >= f.fs_required
    s_ok = worst["settlement_mm"] <= f.settle_allow_mm
    slope_ok = max_slope <= slope_limit
    passed = fs_ok and s_ok and slope_ok
    margin = min(
        worst["fs"] / f.fs_required - 1.0,
        1.0 - worst["settlement_mm"] / f.settle_allow_mm,
        1.0 - max_slope / slope_limit,
    )
    reasons = []
    if not fs_ok:
        reasons.append(
            f"承载力不足：最不利点 FS={worst['fs']:.2f} < {f.fs_required:.1f}"
            f"（qu={worst['qu_kpa']:.0f} kPa，附加应力 {worst['q_applied_kpa']:.0f} kPa）"
        )
    if not s_ok:
        reasons.append(
            f"沉降超限：{worst['settlement_mm']:.1f} mm > {f.settle_allow_mm:.0f} mm"
            f"（E={worst['E']:.1f} MPa）"
        )
    if not slope_ok:
        reasons.append(f"场地坡度 {max_slope:.1f}° 超过 {slope_limit:.0f}°")
    if passed:
        reasons.append(
            f"最不利点 FS={worst['fs']:.2f}，沉降 {worst['settlement_mm']:.1f} mm，"
            f"最大坡度 {max_slope:.1f}°"
        )
    return Verdict(
        key="infrastructure",
        title="基础设施建设可行性",
        passed=passed,
        margin=margin,
        metrics={
            "worst_fs": worst["fs"],
            "worst_settlement_mm": worst["settlement_mm"],
            "max_slope_deg": max_slope,
            "qu_kpa": worst["qu_kpa"],
            "q_applied_kpa": worst["q_applied_kpa"],
            "sites": len(rows),
            "worst_tile": worst["tile"],
        },
        reasons=reasons,
        inputs={"B": f.B, "L": f.L, "D": f.D, "load_kN": f.load_kN,
                "fs_required": f.fs_required, "settle_allow_mm": f.settle_allow_mm},
    )


def verify_slope(world: World, region: Region | None = None, *, depth: float = 1.5,
                 fs_required: float = 1.3) -> Verdict:
    """无限边坡稳定（含地下水位产生的孔隙水压）。默认评估全域最不利坡面。"""
    region = region or Region.whole()
    tiles = world.region_tiles(region)
    worst = None
    for t in tiles:
        beta = math.radians(max(world.slope_deg(t), 0.5))
        col = t.col
        top = col.material_at(0.0)
        prof = col.weighted(0.0, depth, water_table=None)
        gamma = prof["gamma"]
        c, phi = prof["c"], prof["phi"]
        phi_r = math.radians(phi)
        u = 0.0
        if t.wt < depth:
            u = GAMMA_W * (depth - t.wt) * math.cos(beta) ** 2
        drive = gamma * depth * math.sin(beta) * math.cos(beta)
        resist = c + (gamma * depth * math.cos(beta) ** 2 - u) * math.tan(phi_r)
        fs = resist / max(drive, 1e-6)
        if worst is None or fs < worst[0]:
            worst = (fs, t, beta)
    fs, t, beta = worst
    passed = fs >= fs_required
    return Verdict(
        key="slope",
        title="边坡稳定性",
        passed=passed,
        margin=fs / fs_required - 1.0,
        metrics={"worst_fs": fs, "slope_deg": math.degrees(beta),
                 "water_table_m": t.wt, "tile": (t.ix, t.iy)},
        reasons=[f"最不利坡面 FS={fs:.2f}" + ("" if passed else f" < {fs_required:.1f}")],
        inputs={"depth_m": depth, "fs_required": fs_required},
    )


# ---------------------------------------------------------------------------
# 目的 B：着陆
# ---------------------------------------------------------------------------
@dataclass
class Lander:
    mass_kg: float = 1200.0       # 着陆器质量
    n_pads: int = 4
    pad_diameter: float = 0.6     # 足垫直径 m
    touchdown_v: float = 2.5      # 触地速度 m/s
    leg_stroke: float = 0.35      # 缓冲行程 m
    fs_required: float = 2.0
    max_decel_g: float = 12.0
    slope_limit: float = 10.0


def verify_landing(
    world: World, region: Region, lander: Lander | None = None,
    *, n_sites: int = 6,
) -> Verdict:
    """着陆可行性：足垫接触压力 vs 风化层承载力。"""
    ld = lander or Lander()
    g = world.config.gravity
    tiles = world.region_tiles(region)
    if not tiles:
        raise ValueError("区域内没有区块")
    soft = sorted(tiles, key=lambda t: t.col.material_at(0.0).E)[:4]
    step = max(1, len(tiles) // n_sites)
    merged = {t.ix * 10000 + t.iy: t for t in soft + tiles[::step][:n_sites]}
    sites = list(merged.values())

    area = math.pi * ld.pad_diameter**2 / 4.0
    p_static = ld.mass_kg * g / (ld.n_pads * area) / 1000.0        # kPa
    a_decel = ld.touchdown_v**2 / (2 * max(ld.leg_stroke, 1e-6))
    daf = 1.0 + a_decel / max(g, 1e-6)
    p_dyn = p_static * daf

    rows = []
    qu_fn = solvers.property_solver(
        "bearing_qu",
        lambda p: bearing_capacity(p["c"], p["phi"], p["gamma"], p["B"],
                                   p["D"], p["shape"], p["L"]),
    )
    settle_fn = solvers.property_solver(
        "settlement_mm",
        lambda p: elastic_settlement(p["q"], p["B"], p["E"], p["nu"], p["shape"]),
    )
    for t in sites:
        gamma_scale = g / G_EARTH
        prof = t.col.weighted(0.0, max(ld.pad_diameter, 0.5), water_table=t.wt)
        gamma = prof["gamma"] * gamma_scale
        bc = solvers.normalize_capacity(qu_fn({
            "c": prof["c"], "phi": prof["phi"], "gamma": gamma,
            "B": ld.pad_diameter, "D": 0.0, "shape": "circle",
            "L": ld.pad_diameter, "gravity": g,
        }))
        fs = bc["qu_kpa"] / max(p_dyn, 1e-9)
        pen = float(settle_fn({"q": p_dyn, "B": ld.pad_diameter,
                               "E": prof["E"], "nu": prof["nu"], "shape": "circle"}))
        rows.append({"tile": (t.ix, t.iy), "qu": bc["qu_kpa"], "fs": fs,
                     "pen_mm": pen, "c": prof["c"], "phi": prof["phi"],
                     "E": prof["E"], "slope": world.slope_deg(t)})
    worst = min(rows, key=lambda r: r["fs"])
    fs_ok = worst["fs"] >= ld.fs_required
    pen_ok = worst["pen_mm"] <= ld.leg_stroke * 1000 * 0.5
    decel_ok = a_decel / G_EARTH <= ld.max_decel_g
    slope_ok = max(r["slope"] for r in rows) <= ld.slope_limit
    passed = fs_ok and pen_ok and decel_ok and slope_ok

    margin = min(
        worst["fs"] / ld.fs_required - 1.0,
        1.0 - worst["pen_mm"] / (ld.leg_stroke * 500),
        1.0 - (a_decel / G_EARTH) / ld.max_decel_g,
    )
    reasons = []
    if not fs_ok:
        reasons.append(
            f"足垫承载不足：FS={worst['fs']:.2f} < {ld.fs_required:.1f}"
            f"（qu={worst['qu']:.1f} kPa，动力接触压力 {p_dyn:.1f} kPa）"
        )
    if not pen_ok:
        reasons.append(
            f"穿透/沉降 {worst['pen_mm']:.0f} mm 超过缓冲行程的一半"
            f"（{ld.leg_stroke*500:.0f} mm）"
        )
    if not decel_ok:
        reasons.append(f"峰值减速度 {a_decel/G_EARTH:.1f} g 超过限值 {ld.max_decel_g:.0f} g")
    if not slope_ok:
        reasons.append(f"着陆场坡度超过 {ld.slope_limit:.0f}°")
    if passed:
        reasons.append(
            f"FS={worst['fs']:.2f}，穿透 {worst['pen_mm']:.0f} mm，"
            f"减速度 {a_decel/G_EARTH:.1f} g（局部重力 {g:.2f} m/s²）"
        )
    return Verdict(
        key="landing",
        title="着陆可行性",
        passed=passed,
        margin=margin,
        metrics={
            "worst_fs": worst["fs"],
            "qu_kpa": worst["qu"],
            "contact_pressure_static_kpa": p_static,
            "contact_pressure_dynamic_kpa": p_dyn,
            "penetration_mm": worst["pen_mm"],
            "decel_g": a_decel / G_EARTH,
            "local_gravity": g,
            "worst_tile": worst["tile"],
            "worst_slope_deg": max(r["slope"] for r in rows),
            "worst_c_kpa": worst["c"],
            "worst_phi_deg": worst["phi"],
        },
        reasons=reasons,
        inputs={"mass_kg": ld.mass_kg, "n_pads": ld.n_pads,
                "pad_diameter_m": ld.pad_diameter, "touchdown_v": ld.touchdown_v,
                "leg_stroke_m": ld.leg_stroke, "fs_required": ld.fs_required},
    )


# ---------------------------------------------------------------------------
# 目的 C：生态可持续性
# ---------------------------------------------------------------------------
def verify_ecology(world: World, region: Region | None = None,
                   *, animal_stats: dict | None = None,
                   min_vegetated: float = 0.6, min_biomes: int = 2) -> Verdict:
    """生物群落是否在该区块上可持续：植被覆盖、群区多样性、营养级容量。"""
    region = region or Region.whole()
    tiles = world.region_tiles(region)
    veg = [t for t in tiles if t.biomass > 0.15]
    frac = len(veg) / max(len(tiles), 1)

    biomes: dict[str, int] = {}
    for t in veg:
        biomes[t.biome] = biomes.get(t.biome, 0) + 1
    p = np.array(list(biomes.values()), dtype=float)
    p = p / p.sum() if p.size else p
    shannon = float(-(p * np.log(np.maximum(p, 1e-12))).sum()) if p.size else 0.0

    mean_npp = float(np.mean([t.npp for t in tiles]))
    mean_bio = float(np.mean([t.biomass for t in tiles]))
    woody = sum(1 for t in veg if t.cohort and
                any(cc.height > 3 for cc in t.cohort)) / max(len(veg), 1)

    animals = animal_stats or {}
    richness = len(animals.get("by_species", {}))
    largest = animals.get("largest_survivor_kg", 0.0)

    veg_ok = frac >= min_vegetated and mean_npp > 0.05
    biome_ok = len(biomes) >= min_biomes
    animal_ok = (not animal_stats) or (richness >= 1 and animals.get("population", 0) > 20)
    passed = veg_ok and biome_ok and animal_ok

    margin = min(frac / min_vegetated - 1.0,
                 len(biomes) / min_biomes - 1.0,
                 1.0 if animal_ok else -1.0)
    reasons = []
    if not veg_ok:
        reasons.append(f"植被覆盖仅 {frac*100:.0f}%，平均 NPP {mean_npp:.3f} kg C/m²/yr")
    if not biome_ok:
        reasons.append(f"仅 {len(biomes)} 个生物群区，低于下限 {min_biomes}")
    if animal_stats and not animal_ok:
        reasons.append(f"动物种群 {animals.get('population', 0)} 只，不足以维持营养级")
    if passed:
        reasons.append(
            f"植被覆盖 {frac*100:.0f}%，{len(biomes)} 个群区（Shannon {shannon:.2f}），"
            f"平均 NPP {mean_npp:.2f}"
        )
    return Verdict(
        key="ecology",
        title="生物群落可持续性",
        passed=passed,
        margin=margin,
        metrics={
            "vegetated_fraction": frac,
            "biomes": biomes,
            "shannon": shannon,
            "mean_npp": mean_npp,
            "mean_biomass": mean_bio,
            "woody_fraction": woody,
            "animal_population": animals.get("population", 0),
            "animal_richness": richness,
            "largest_survivor_kg": largest,
        },
        reasons=reasons,
        inputs={"min_vegetated": min_vegetated, "min_biomes": min_biomes},
    )


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def evaluate_all(world: World, *, site: Region | None = None,
                 foundation: Foundation | None = None,
                 lander: Lander | None = None,
                 animal_stats: dict | None = None) -> list[Verdict]:
    site = site or Region.whole()
    out = [
        verify_infrastructure(world, site, foundation),
        verify_slope(world, Region.whole()),
        verify_landing(world, site, lander),
        verify_ecology(world, Region.whole(), animal_stats=animal_stats),
    ]
    return out
