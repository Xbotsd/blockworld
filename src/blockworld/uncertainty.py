"""不确定性与全局敏感性。

为什么要这个模块：上一版用单点参数扫描，秩相关必然饱和到 ±1，那个"敏感性排序"
没有信息量。这里换成两件真正能回答问题的事：

  * Monte Carlo：给定参数分布，输出**失效概率** P(fail) 与裕度分位数，
    而不是一个看起来精确的安全系数；
  * Morris 筛选：在参数空间里做一次轨迹设计，得到 mu*（影响强度）与 sigma
    （非线性/交互强度），用几十次评估就能排出"哪些参数值得精细标定"。

所有随机性都来自显式 seed，结果可复现。
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from . import materials
from .api import (Intervention, Study, choose_foundation_site,
                  choose_landing_site)
from .world import Region, World


# ---------------------------------------------------------------------------
# 分布
# ---------------------------------------------------------------------------
@dataclass
class Distribution:
    kind: str = "uniform"          # uniform | triangular | normal | lognormal
    low: float = 0.0
    high: float = 1.0
    mode: float | None = None
    mean: float | None = None
    sd: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("uniform", "triangular", "normal", "lognormal"):
            raise ValueError(f"未知分布 {self.kind!r}")
        if self.kind in ("uniform", "triangular") and self.high <= self.low:
            raise ValueError("区间分布要求 high > low")
        if self.kind == "triangular" and not (self.low <= (self.mode if self.mode is not None else self.low) <= self.high):
            raise ValueError("三角分布的众数必须落在 [low, high]")

    def sample(self, rng: np.random.Generator) -> float:
        if self.kind == "uniform":
            return float(rng.uniform(self.low, self.high))
        if self.kind == "triangular":
            m = self.mode if self.mode is not None else 0.5 * (self.low + self.high)
            return float(rng.triangular(self.low, m, self.high))
        mu = self.mean if self.mean is not None else 0.5 * (self.low + self.high)
        sd = self.sd if self.sd is not None else 0.25 * (self.high - self.low)
        if self.kind == "normal":
            return float(rng.normal(mu, sd))
        return float(rng.lognormal(math.log(max(mu, 1e-9)), max(sd / max(mu, 1e-9), 1e-6)))

    def unit_to_value(self, u: float) -> float:
        """把 [0,1] 上的位置映射到取值（Morris 轨迹设计用）。"""
        u = min(max(u, 0.0), 1.0)
        if self.kind == "uniform":
            return self.low + u * (self.high - self.low)
        if self.kind == "triangular":
            m = self.mode if self.mode is not None else 0.5 * (self.low + self.high)
            # 反 CDF
            F = (m - self.low) / (self.high - self.low)
            if u <= F:
                return self.low + math.sqrt(u * (self.high - self.low) * (m - self.low))
            return self.high - math.sqrt((1 - u) * (self.high - self.low) * (self.high - m))
        mu = self.mean if self.mean is not None else 0.5 * (self.low + self.high)
        sd = self.sd if self.sd is not None else 0.25 * (self.high - self.low)
        if self.kind == "normal":
            from statistics import NormalDist

            return float(NormalDist(mu, sd).inv_cdf(min(max(u, 1e-6), 1 - 1e-6)))
        return float(math.exp(NormalDist(math.log(max(mu, 1e-9)),
                                         max(sd / max(mu, 1e-9), 1e-6))
                              .inv_cdf(min(max(u, 1e-6), 1 - 1e-6))))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "low": self.low, "high": self.high,
                "mode": self.mode, "mean": self.mean, "sd": self.sd}


# ---------------------------------------------------------------------------
# 参数：名字 + 分布 + 作用目标
# ---------------------------------------------------------------------------
@dataclass
class Parameter:
    """一个不确定输入。

    target 是声明式的作用位置（可序列化），目前支持：

      {"kind": "material_prop", "material": "lunar_regolith", "prop": "c",
       "region": {...}}                        # 区域内的某材料参数
      {"kind": "gravity"}                      # 局部重力
      {"kind": "foundation", "field": "B"}     # 判据输入
      {"kind": "lander", "field": "pad_diameter"}
      {"kind": "precip_factor", "region": {...}}   # 降水倍数
    """

    name: str
    target: dict
    dist: Distribution
    label: str = ""

    def describe(self) -> str:
        return self.label or self.name

    def apply(self, world: World, criteria):
        """把抽样值应用到世界/判据；返回（可能被替换的）criteria。"""
        v = getattr(self, "_value", None)
        if v is None:
            raise RuntimeError("先设置 _value 再 apply")
        kind = self.target["kind"]
        if kind == "material_prop":
            region = (Region.from_dict(self.target["region"])
                      if "region" in self.target else Region.whole())
            key, prop = self.target["material"], self.target["prop"]
            for t in world.region_tiles(region):
                new_layers = []
                for th, m in t.col.layers:
                    if m.key == key:
                        m2 = m.with_params(**{prop: v})
                        errs, _ = m2.check()
                        if errs:
                            raise materials.DomainError("；".join(errs))
                        new_layers.append((th, m2))
                    else:
                        new_layers.append((th, m))
                t.col.layers = new_layers
        elif kind == "gravity":
            world.config.gravity = float(v)
        elif kind == "precip_factor":
            region = (Region.from_dict(self.target["region"])
                      if "region" in self.target else Region.whole())
            for t in world.region_tiles(region):
                t.precip = float(t.precip * v)
        elif kind == "foundation":
            setattr(criteria.foundation, self.target["field"], float(v))
        elif kind == "lander":
            setattr(criteria.lander, self.target["field"], float(v))
        else:
            raise ValueError(f"未知参数目标 {kind!r}")
        return criteria

    def to_dict(self) -> dict:
        return {"name": self.name, "label": self.label, "target": self.target,
                "dist": self.dist.to_dict()}


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
@dataclass
class MonteCarloResult:
    n: int
    params: list[dict]
    baseline: dict            # criterion -> {p_fail, margin_p05/p50/p95}
    intervention: dict | None
    intervention_key: str | None = None
    samples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n": self.n, "params": self.params,
            "baseline": self.baseline,
            "intervention": self.intervention,
            "intervention_key": self.intervention_key,
            "samples": self.samples[:200],
        }


def _summarize(records: list[dict]) -> dict:
    """按判据汇总：失效概率 + 裕度分位数。"""
    by_key: dict[str, list[float]] = {}
    fails: dict[str, int] = {}
    for rec in records:
        for key, margin in rec.items():
            by_key.setdefault(key, []).append(margin)
            fails[key] = fails.get(key, 0) + (1 if margin < 0 else 0)
    out = {}
    for key, margins in by_key.items():
        arr = np.array(margins)
        out[key] = {
            "n": int(arr.size),
            "p_fail": round(fails.get(key, 0) / max(arr.size, 1), 4),
            "margin_p05": round(float(np.percentile(arr, 5)), 4),
            "margin_p50": round(float(np.percentile(arr, 50)), 4),
            "margin_p95": round(float(np.percentile(arr, 95)), 4),
        }
    return out


def monte_carlo(
    study: Study,
    params: list[Parameter],
    *,
    n: int = 100,
    seed: int = 0,
    intervention: Intervention | None = None,
) -> MonteCarloResult:
    """对基线（及一次干预）做 Monte Carlo，返回失效概率。"""
    if n < 2:
        raise ValueError("n 至少为 2")
    rng = np.random.default_rng(seed)
    world0 = study._prepare()
    sites = {
        "foundation": study.criteria.foundation_site or choose_foundation_site(world0),
        "landing": study.criteria.landing_site or choose_landing_site(world0),
    }

    base_records: list[dict] = []
    iv_records: list[dict] = []
    samples: list[dict] = []

    for i in range(n):
        values = {}
        for p in params:
            p._value = p.dist.sample(rng)
            values[p.name] = round(float(p._value), 6)

        world = copy.deepcopy(world0)
        criteria = copy.deepcopy(study.criteria)
        for p in params:
            criteria = p.apply(world, criteria)
        base_verdicts = study.evaluate(world, sites, criteria=criteria)
        base_records.append({v.key: v.margin for v in base_verdicts})

        if intervention is not None:
            world2 = copy.deepcopy(world)
            intervention.apply(world2)
            iv_verdicts = study.evaluate(world2, sites, criteria=criteria)
            iv_records.append({v.key: v.margin for v in iv_verdicts})

        samples.append(values)

    return MonteCarloResult(
        n=n,
        params=[p.to_dict() for p in params],
        baseline=_summarize(base_records),
        intervention=_summarize(iv_records) if intervention is not None else None,
        intervention_key=intervention.key if intervention is not None else None,
        samples=samples,
    )


# ---------------------------------------------------------------------------
# Morris 筛选（初等效应）
# ---------------------------------------------------------------------------
@dataclass
class MorrisResult:
    r: int
    levels: int
    evaluations: int
    params: list[dict]
    baseline: dict            # criterion -> {param: {mu, mu_star, sigma, rank}}
    intervention: dict | None
    intervention_key: str | None = None

    def ranking(self, criterion: str, *, use: str = "intervention") -> list[dict]:
        table = self.intervention if (use == "intervention" and self.intervention) else self.baseline
        rows = [{"param": k, **v} for k, v in table.get(criterion, {}).items()]
        return sorted(rows, key=lambda d: -d["mu_star"])

    def to_dict(self) -> dict:
        return {"r": self.r, "levels": self.levels, "evaluations": self.evaluations,
                "params": self.params, "baseline": self.baseline,
                "intervention": self.intervention,
                "intervention_key": self.intervention_key}


def morris(
    study: Study,
    params: list[Parameter],
    *,
    r: int = 6,
    levels: int = 4,
    seed: int = 0,
    intervention: Intervention | None = None,
    max_evaluations: int = 400,
) -> MorrisResult:
    """Morris 轨迹设计：每个参数得到 mu（方向）、mu*（强度）、sigma（非线性/交互）。"""
    p = len(params)
    if p == 0:
        raise ValueError("至少需要一个参数")
    if levels < 2:
        raise ValueError("levels 至少为 2")
    evals_needed = r * (p + 1)
    if evals_needed > max_evaluations:
        raise ValueError(
            f"评估次数 {evals_needed} 超过上限 {max_evaluations}；"
            f"减少 r / 参数个数，或在预算内调大 max_evaluations"
        )

    rng = np.random.default_rng(seed)
    delta = levels / (2.0 * (levels - 1))
    world0 = study._prepare()
    from .api import choose_foundation_site, choose_landing_site

    sites = {
        "foundation": study.criteria.foundation_site or choose_foundation_site(world0),
        "landing": study.criteria.landing_site or choose_landing_site(world0),
    }

    ee_base: dict[str, dict[str, list[float]]] = {pp.name: {} for pp in params}
    ee_iv: dict[str, dict[str, list[float]]] = {pp.name: {} for pp in params}
    base_out: list[dict] = []
    iv_out: list[dict] = []

    def evaluate(u: list[float], iv: bool) -> dict:
        world = copy.deepcopy(world0)
        criteria = copy.deepcopy(study.criteria)
        for k, p_ in enumerate(params):
            p_._value = p_.dist.unit_to_value(u[k])
            criteria = p_.apply(world, criteria)
        if iv and intervention is not None:
            intervention.apply(world)
        verdicts = study.evaluate(world, sites, criteria=criteria)
        return {v.key: v.margin for v in verdicts}

    for traj in range(r):
        grid = [k / (levels - 1) for k in range(levels)]
        starts = [g for g in grid if g <= 1.0 - delta + 1e-9]
        u = [float(rng.choice(starts)) for _ in range(p)]
        order = list(rng.permutation(p))

        cur = evaluate(u, False)
        cur_iv = evaluate(u, True) if intervention is not None else None
        for idx in order:
            u2 = list(u)
            step = delta if u2[idx] + delta <= 1.0 else -delta
            u2[idx] = u2[idx] + step
            nxt = evaluate(u2, False)
            nxt_iv = evaluate(u2, True) if intervention is not None else None
            for key, val in cur.items():
                d = (nxt.get(key, 0.0) - val) / (step if step != 0 else 1e-9)
                ee_base[params[idx].name].setdefault(key, []).append(d)
            if cur_iv is not None:
                for key, val in cur_iv.items():
                    d = (nxt_iv.get(key, 0.0) - val) / (step if step != 0 else 1e-9)
                    ee_iv[params[idx].name].setdefault(key, []).append(d)
            u = u2
            cur = nxt
            cur_iv = nxt_iv

    def package(ee: dict[str, list[dict]]) -> dict:
        out: dict[str, dict] = {}
        for name, per_key in ee.items():
            for key, values in per_key.items():
                arr = np.array(values, dtype=float)
                out.setdefault(key, {})[name] = {
                    "mu": round(float(arr.mean()), 4),
                    "mu_star": round(float(np.abs(arr).mean()), 4),
                    "sigma": round(float(arr.std(ddof=1)) if arr.size > 1 else 0.0, 4),
                }
        for key, table in out.items():
            ranked = sorted(table.items(), key=lambda kv: -kv[1]["mu_star"])
            for rank, (name, row) in enumerate(ranked, start=1):
                row["rank"] = rank
        return out

    return MorrisResult(
        r=r, levels=levels, evaluations=evals_needed,
        params=[p.to_dict() for p in params],
        baseline=package(ee_base),
        intervention=package(ee_iv) if intervention is not None else None,
        intervention_key=intervention.key if intervention is not None else None,
    )
