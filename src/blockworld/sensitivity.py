"""敏感性与判据翻转分析。

替换本身不是目的，"替换后结论会不会翻转"才是。这一层做三件事：
  1. 单因素（OAT）扫描：一次只改一个参数，看判据裕度怎么走；
  2. 秩相关排序：给出"哪个参数真正在主导结论"，用于决定值得精细标定的少数参数；
  3. 翻转点定位：在扫描序列里找出通过/不通过的边界，并给出临界参数值。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import materials
from .verifiers import Verdict
from .world import Region, World


# ---------------------------------------------------------------------------
# 秩相关
# ---------------------------------------------------------------------------
def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    # 处理并列：取平均秩
    vals, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(vals))
        np.add.at(sums, inv, ranks)
        means = sums / counts
        ranks = means[inv]
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    a, b = np.asarray(x, float), np.asarray(y, float)
    if a.size < 2 or b.size < 2 or np.allclose(a, a[0]):
        return 0.0
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# 扫描
# ---------------------------------------------------------------------------
@dataclass
class SweepRow:
    value: float
    metric: float
    passed: bool
    margin: float
    notes: list[str] = field(default_factory=list)


@dataclass
class SweepResult:
    factor: str
    label: str
    rows: list[SweepRow]

    @property
    def rho(self) -> float:
        return spearman([r.value for r in self.rows], [r.metric for r in self.rows])

    @property
    def flips(self) -> list[dict]:
        out = []
        for a, b in zip(self.rows, self.rows[1:]):
            if a.passed != b.passed:
                # 线性插值出临界值
                dm = b.metric - a.metric
                t = 0.0 if abs(dm) < 1e-12 else (0.0 - a.metric) / dm
                t = min(max(t, 0.0), 1.0)
                out.append({
                    "from": a.value, "to": b.value,
                    "critical": round(a.value + t * (b.value - a.value), 4),
                    "direction": "失效" if a.passed else "恢复",
                })
        return out

    def to_dict(self) -> dict:
        return {
            "factor": self.factor,
            "label": self.label,
            "rho": round(self.rho, 3),
            "flips": self.flips,
            "rows": [
                {"value": r.value, "metric": round(r.metric, 4),
                 "passed": r.passed, "margin": round(r.margin, 4)}
                for r in self.rows
            ],
        }


def oat_simple(
    base: World,
    name: str,
    label: str,
    values: list[float],
    apply: Callable[[World, float], None],
    evaluate: Callable[[World], Verdict],
    metric_key: str = "worst_fs",
) -> SweepResult:
    rows: list[SweepRow] = []
    for v in values:
        w = copy.deepcopy(base)
        apply(w, v)
        verdict = evaluate(w)
        metric = float(verdict.metrics.get(metric_key, verdict.margin))
        rows.append(SweepRow(float(v), metric, verdict.passed, verdict.margin,
                             list(verdict.reasons)))
    return SweepResult(name, label, rows)


def rank(results: list[SweepResult]) -> list[dict]:
    """按 |Spearman ρ| 排序，给出主导参数。"""
    rows = [
        {"factor": r.factor, "label": r.label, "rho": round(r.rho, 3),
         "abs_rho": round(abs(r.rho), 3), "flips": len(r.flips)}
        for r in results
    ]
    return sorted(rows, key=lambda d: -d["abs_rho"])


# ---------------------------------------------------------------------------
# 二维翻转面
# ---------------------------------------------------------------------------
def grid_2d(
    base: World,
    xs: list[float],
    ys: list[float],
    apply: Callable[[World, float, float], None],
    evaluate: Callable[[World], Verdict],
    metric_key: str = "worst_fs",
) -> dict:
    z, passed = [], []
    for y in ys:
        row_z, row_p = [], []
        for x in xs:
            w = copy.deepcopy(base)
            apply(w, x, y)
            v = evaluate(w)
            row_z.append(round(float(v.metrics.get(metric_key, v.margin)), 3))
            row_p.append(v.passed)
        z.append(row_z)
        passed.append(row_p)
    return {"x": xs, "y": ys, "z": z, "passed": passed}


# ---------------------------------------------------------------------------
# 常用施动函数
# ---------------------------------------------------------------------------
def set_material_property(world: World, region: Region, material_key: str,
                          prop: str, value: float) -> None:
    """把区域内指定材料的某个参数替换为给定值（其余参数不变）。"""
    for t in world.region_tiles(region):
        new_layers = []
        for th, m in t.col.layers:
            if m.key == material_key:
                m2 = m.with_params(**{prop: value})
                errors, _ = m2.check()
                if errors:
                    raise materials.DomainError("；".join(errors))
                new_layers.append((th, m2))
            else:
                new_layers.append((th, m))
        t.col.layers = new_layers


def set_gravity(world: World, value: float) -> None:
    if not 0 < value <= 30:
        raise materials.DomainError(f"重力 {value} 超出范围")
    world.config.gravity = value
