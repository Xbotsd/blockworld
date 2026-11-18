"""公开 API：把"世界 + 替换 + 判据"包成一个可复用的问题求解接口。

典型用法::

    from blockworld import Study, Criteria, Intervention

    study = Study(
        site=site,                                   # 场地数据（DEM/钻孔/气候）
        criteria=Criteria(purposes=("infrastructure", "landing")),
        interventions=(
            Intervention("换填", [{"op": "material", "region": {...},
                                   "material": "gravel_dense",
                                   "top_depth": 0, "bottom_depth": 8}]),
        ),
    )
    result = study.run()
    result.passed()          # {"baseline": {...}, "换填": {...}}
    result.to_dict()         # 可 JSON 序列化，进报告/入库/比对

设计约束（这是它区别于 demo 的地方）：
  * 不隐藏数据来源：``site.provenance`` 会写进结果；
  * 不静默失败：非法替换抛异常，而不是产生一个"能算但没有物理意义"的世界；
  * 每一次判据都带 margin，便于比较与做不确定性分析。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import edits, hydrology, solvers, vegetation
from . import verifiers as V
from .site import SiteSpec, build_from
from .world import Region, World


# ---------------------------------------------------------------------------
# 判据配置
# ---------------------------------------------------------------------------
@dataclass
class Criteria:
    """要验证哪些目的，以及各自的阈值。"""

    purposes: tuple[str, ...] = ("infrastructure", "slope", "landing", "ecology")
    foundation: V.Foundation = field(default_factory=V.Foundation)
    lander: V.Lander = field(default_factory=V.Lander)
    slope_fs_required: float = 1.3
    slope_limit_deg: float = 20.0
    eco_min_vegetated: float = 0.6
    eco_min_biomes: int = 2
    eco_animal_years: int = 0                # >0 时在生态判据前跑动物推演
    foundation_site: Region | None = None    # None → 自动选址
    landing_site: Region | None = None

    def to_dict(self) -> dict:
        return {
            "purposes": list(self.purposes),
            "foundation": vars(self.foundation),
            "lander": vars(self.lander),
            "slope_fs_required": self.slope_fs_required,
            "slope_limit_deg": self.slope_limit_deg,
            "eco_min_vegetated": self.eco_min_vegetated,
            "eco_min_biomes": self.eco_min_biomes,
            "eco_animal_years": self.eco_animal_years,
        }


# ---------------------------------------------------------------------------
# 选址策略（明确写出来，而不是藏在 demo 里）
# ---------------------------------------------------------------------------
def choose_foundation_site(world: World, *, max_slope: float = 8.0,
                           max_elev: float = 60.0, radius: float = 200.0) -> Region:
    """基础设施选址：缓坡低地中最软的地基（"最不利点"语义）。"""
    cands = [t for t in world.tiles
             if world.slope_deg(t) < max_slope and t.elev < max_elev]
    if not cands:
        cands = world.tiles
    t = min(cands, key=lambda x: (x.col.material_at(1.5).E, x.col.material_at(1.5).c))
    return Region.circle(t.x, t.y, radius)


def choose_landing_site(world: World, *, max_elev: float = 120.0,
                        radius: float = 150.0, reach: int = 1) -> Region:
    """着陆选址：低地中最平坦的区域（邻域内最大坡度最小）。"""
    best = None
    for t in world.tiles:
        if t.elev > max_elev:
            continue
        nb = [t]
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                jx, jy = t.ix + dx, t.iy + dy
                if 0 <= jx < world.config.nx and 0 <= jy < world.config.ny:
                    nb.append(world.tile(jx, jy))
        m = max(world.slope_deg(x) for x in nb)
        if best is None or m < best[0]:
            best = (m, t)
    return Region.circle(best[1].x, best[1].y, radius)


# ---------------------------------------------------------------------------
# 干预（可序列化）
# ---------------------------------------------------------------------------
@dataclass
class Intervention:
    """一次可替换的工程/管理措施，由若干原子操作组成。"""

    key: str
    ops: tuple[dict, ...]
    description: str = ""

    def apply(self, world: World) -> edits.EditSession:
        session = edits.EditSession(world)
        for op in self.ops:
            _apply_op(session, op)
        return session

    def to_dict(self) -> dict:
        return {"key": self.key, "description": self.description,
                "ops": [dict(o) for o in self.ops]}

    @staticmethod
    def from_dict(d: dict) -> "Intervention":
        return Intervention(d["key"], tuple(d.get("ops", ())), d.get("description", ""))


def _apply_op(session: edits.EditSession, op: dict) -> None:
    kind = op["op"]
    region = Region.from_dict(op["region"]) if "region" in op else Region.whole()
    if kind == "material":
        session.set_material(
            region, op["material"], top_depth=float(op.get("top_depth", 0.0)),
            bottom_depth=op.get("bottom_depth"), model=op.get("model"),
        )
    elif kind == "elevation":
        session.set_elevation(region, delta=op.get("delta"), absolute=op.get("absolute"))
    elif kind == "water_table":
        session.set_water_table(region, float(op["depth"]))
    elif kind == "precip":
        session.set_precip(region, factor=op.get("factor"), absolute=op.get("absolute"))
    elif kind == "pft":
        session.set_pft(region, op["pft"], mode=op.get("mode", "replace"))
    elif kind == "gravity":
        session.set_gravity(float(op["value"]))
    else:
        raise ValueError(f"未知操作 {kind!r}；可用：material/elevation/water_table/"
                         f"precip/pft/gravity")


# ---------------------------------------------------------------------------
# 研究：一次完整的"基线 → 干预 → 判据"
# ---------------------------------------------------------------------------
@dataclass
class Study:
    site: SiteSpec
    criteria: Criteria = field(default_factory=Criteria)
    interventions: tuple[Intervention, ...] = ()
    spin_years: int = 0          # >0 时先跑植被-水文平衡
    seed: int = 0

    # -- 内部 -----------------------------------------------------------
    def _prepare(self) -> World:
        world = build_from(self.site)
        if self.spin_years > 0:
            hydrology.step(world)
            vegetation.spin_up(world, years=self.spin_years)
        return world

    def sites(self, world: World) -> dict:
        c = self.criteria
        return {
            "foundation": c.foundation_site or choose_foundation_site(world),
            "landing": c.landing_site or choose_landing_site(world),
        }

    def evaluate(self, world: World, sites: dict, *,
                 criteria: Criteria | None = None,
                 animal_stats: dict | None = None) -> list[V.Verdict]:
        return evaluate(world, criteria or self.criteria, sites=sites,
                        animal_stats=animal_stats)

    # -- 公开 -----------------------------------------------------------
    def run(self) -> "StudyResult":
        base_world = self._prepare()
        sites = self.sites(base_world)
        c = self.criteria

        animal_stats = None
        if "ecology" in c.purposes and c.eco_animal_years > 0:
            from . import animals

            animal_stats = animals.run(base_world, years=c.eco_animal_years,
                                       seed=self.seed + 11, initial=200)

        baseline = self.evaluate(base_world, sites, animal_stats=animal_stats)

        results: dict[str, dict] = {}
        for iv in self.interventions:
            world = self._prepare()
            session = iv.apply(world)
            if self.spin_years > 0:
                hydrology.step(world)
                vegetation.spin_up(world, years=max(10, self.spin_years // 3))
            stats = animal_stats
            if stats is not None:
                from . import animals

                stats = animals.run(world, years=max(2, c.eco_animal_years // 2),
                                    seed=self.seed + 12, initial=150)
            verdicts = self.evaluate(world, sites, animal_stats=stats)
            results[iv.key] = {
                "description": iv.description,
                "verdicts": [v.to_dict() for v in verdicts],
                "log": session.log(),
                "deltas": _deltas(baseline, verdicts),
            }

        return StudyResult(
            baseline=[v.to_dict() for v in baseline],
            interventions=results,
            meta={
                "source": self.site.provenance.get("source", "user-data"),
                "provenance": self.site.provenance,
                "grid": self.site.grid.to_dict(),
                "gravity": self.site.gravity,
                "spin_years": self.spin_years,
                "seed": self.seed,
                "solvers": solvers.active(),
                "foundation_site": sites["foundation"].to_dict(),
                "landing_site": sites["landing"].to_dict(),
                "criteria": c.to_dict(),
                "world": base_world.extent(),
            },
        )


def _deltas(baseline: list[V.Verdict], after: list[V.Verdict]) -> list[dict]:
    by_key = {v.key: v for v in baseline}
    out = []
    for v in after:
        b = by_key.get(v.key)
        if b is None:
            continue
        out.append({
            "criterion": v.key,
            "title": v.title,
            "passed_before": b.passed,
            "passed_after": v.passed,
            "flip": b.passed != v.passed,
            "margin_before": round(b.margin, 4),
            "margin_after": round(v.margin, 4),
            "margin_delta": round(v.margin - b.margin, 4),
        })
    return out


_DEFAULT_CRITERIA = {
    "infrastructure": V.verify_infrastructure,
    "slope": V.verify_slope,
    "landing": V.verify_landing,
    "ecology": V.verify_ecology,
}


def evaluate(world: World, criteria: Criteria, *,
             sites: dict | None = None,
             animal_stats: dict | None = None) -> list[V.Verdict]:
    """按判据评估一个世界；不传 sites 时按默认策略选址。"""
    sites = sites or {
        "foundation": criteria.foundation_site or choose_foundation_site(world),
        "landing": criteria.landing_site or choose_landing_site(world),
    }
    out = []
    for purpose in criteria.purposes:
        fn = solvers.criterion_solver(purpose, _DEFAULT_CRITERIA[purpose])
        if purpose == "infrastructure":
            out.append(fn(world, sites["foundation"], criteria.foundation,
                          slope_limit=criteria.slope_limit_deg))
        elif purpose == "slope":
            out.append(fn(world, Region.whole(),
                          fs_required=criteria.slope_fs_required))
        elif purpose == "landing":
            out.append(fn(world, sites["landing"], criteria.lander))
        elif purpose == "ecology":
            out.append(fn(world, Region.whole(), animal_stats=animal_stats,
                          min_vegetated=criteria.eco_min_vegetated,
                          min_biomes=criteria.eco_min_biomes))
        else:
            raise ValueError(f"未知目的 {purpose!r}")
    return out


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------
@dataclass
class StudyResult:
    baseline: list[dict]
    interventions: dict[str, dict]
    meta: dict

    def passed(self) -> dict:
        out = {"baseline": {v["key"]: v["passed"] for v in self.baseline}}
        for key, payload in self.interventions.items():
            out[key] = {v["key"]: v["passed"] for v in payload["verdicts"]}
        return out

    def flips(self) -> list[dict]:
        out = []
        for key, payload in self.interventions.items():
            for d in payload["deltas"]:
                if d["flip"]:
                    out.append({"intervention": key, **d})
        return out

    def to_dict(self) -> dict:
        return {"baseline": self.baseline, "interventions": self.interventions,
                "meta": self.meta,
                "passed": self.passed(), "flips": self.flips()}
