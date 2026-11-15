"""地质体构建：钻孔 → 插值 → 分层地质柱（第一层"把地质变成可寻址的输入"）。

真实工作流是"钻孔数据标准化 → 界面插值 → 逐层挂材料属性"。这里实现的是同一
条链路的最小可运行版本：
  * 普通克里金（指数变异函数）插值各分层界面高程；
  * 层内材料用指示克里金/反距离加权投票确定；
  * 每个区块持有一根可寻址的分层柱，替换操作直接作用于柱内的深度区间。

对照分享对话：这一步的产出不是一张图，而是"每个空间位置都持有一组可替换物性
参数"的对象。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import materials


# ---------------------------------------------------------------------------
# 普通克里金
# ---------------------------------------------------------------------------
def _variogram(h: np.ndarray, rng: float, sill: float, nugget: float = 0.0) -> np.ndarray:
    """指数变异函数 γ(h) = n + (s-n)(1 - exp(-3h/r))。"""
    rng = max(rng, 1e-6)
    return nugget + (sill - nugget) * (1.0 - np.exp(-3.0 * h / rng))


def krige(
    points: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    *,
    nugget: float = 0.0,
    range_factor: float = 1.2,
) -> np.ndarray:
    """普通克里金插值。

    points/targets: (n,2) / (m,2) 平面坐标；values: (n,)
    返回 (m,) 插值结果。样本不足时退化为反距离加权。
    """
    P = np.asarray(points, dtype=float)
    V = np.asarray(values, dtype=float)
    T = np.asarray(targets, dtype=float)
    if P.shape[0] == 0:
        raise ValueError("没有可用样本点")
    if P.shape[0] < 3:
        return _idw(P, V, T)

    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    rng = float(np.median(d[np.triu_indices_from(d, 1)])) * range_factor
    sill = float(np.var(V)) or 1.0

    A = np.ones((P.shape[0] + 1, P.shape[0] + 1))
    A[:-1, :-1] = _variogram(d, rng, sill, nugget)
    np.fill_diagonal(A[:-1, :-1], 0.0)
    A[-1, -1] = 0.0
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return _idw(P, V, T)

    dt = np.linalg.norm(P[:, None, :] - T[None, :, :], axis=2)   # (n, m)
    b = np.ones((P.shape[0] + 1, T.shape[0]))
    b[:-1, :] = _variogram(dt, rng, sill, nugget)
    w = A_inv @ b                                                  # (n+1, m)
    return (V[:, None] * w[:-1, :]).sum(axis=0)


def _idw(P: np.ndarray, V: np.ndarray, T: np.ndarray, power: float = 2.0) -> np.ndarray:
    d = np.linalg.norm(P[:, None, :] - T[None, :, :], axis=2)
    d = np.maximum(d, 1e-9)
    w = 1.0 / d**power
    return (w * V[:, None]).sum(axis=0) / w.sum(axis=0)


# ---------------------------------------------------------------------------
# 钻孔与分层柱
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Borehole:
    """一个钻孔：地面高程 + 自上而下的（厚度, 材料）序列。"""

    x: float
    y: float
    ground: float
    layers: tuple[tuple[float, str], ...]
    source: str = "survey"

    def boundary_elevations(self) -> list[float]:
        """各层顶面高程（含地面），自下而上仍按自上而下顺序排列。"""
        elevs, z = [], self.ground
        for thickness, _ in self.layers:
            elevs.append(z)
            z -= thickness
        elevs.append(z)  # 柱底
        return elevs

    def validate(self) -> list[str]:
        issues = []
        for i, (t, key) in enumerate(self.layers):
            if t <= 0:
                issues.append(f"第 {i+1} 层厚度非正：{t}")
            if key not in materials.LIBRARY:
                issues.append(f"第 {i+1} 层材料未知：{key}")
        return issues


@dataclass
class LayeredColumn:
    """一个区块的分层地质柱（可寻址、可替换）。"""

    top: float                                    # 地面高程 m
    layers: list[tuple[float, materials.Material]] = field(default_factory=list)

    # -- 结构 -----------------------------------------------------------
    @property
    def bottom(self) -> float:
        return self.top - sum(t for t, _ in self.layers)

    @property
    def thickness(self) -> float:
        return sum(t for t, _ in self.layers)

    def boundaries(self) -> list[tuple[float, float, materials.Material]]:
        """返回 [(顶深, 底深, 材料), ...]，深度自地面起算。"""
        out, z = [], 0.0
        for t, m in self.layers:
            out.append((z, z + t, m))
            z += t
        return out

    def material_at(self, depth: float) -> materials.Material:
        z = 0.0
        for t, m in self.layers:
            if z <= depth < z + t:
                return m
            z += t
        return self.layers[-1][1]

    def layer_index_at(self, depth: float) -> int:
        z = 0.0
        for i, (t, _) in enumerate(self.layers):
            if z <= depth < z + t:
                return i
            z += t
        return len(self.layers) - 1

    # -- 替换 -----------------------------------------------------------
    def replace(self, top_depth: float, bottom_depth: float, material: materials.Material) -> list[dict]:
        """把 [top_depth, bottom_depth) 区间的材料替换掉，保持总厚度不变。

        返回被替换掉的片段记录（用于形成可追溯的编辑日志）。
        """
        if top_depth < 0 or bottom_depth <= top_depth:
            raise materials.DomainError(
                f"替换区间非法：{top_depth:.2f}–{bottom_depth:.2f} m"
            )
        if bottom_depth > self.thickness + 1e-9:
            raise materials.DomainError(
                f"替换区间 {bottom_depth:.2f} m 超出柱底 {self.thickness:.2f} m"
            )
        removed: list[dict] = []
        new_layers: list[tuple[float, materials.Material]] = []
        z = 0.0
        for t, m in self.layers:
            a, b = z, z + t
            if b <= top_depth or a >= bottom_depth:      # 不受影响
                new_layers.append((t, m))
            else:
                lo, hi = max(a, top_depth), min(b, bottom_depth)
                if lo > a:
                    new_layers.append((lo - a, m))
                if lo > a or hi < b:
                    removed.append({"material": m.key, "thickness": hi - lo,
                                    "top_depth": lo, "bottom_depth": hi})
                if hi < b:
                    new_layers.append((b - hi, m))
            z = b
        # 插入替换段
        inserted = (material, bottom_depth - top_depth)
        out: list[tuple[float, materials.Material]] = []
        z = 0.0
        placed = False
        for t, m in new_layers:
            if not placed and z >= top_depth - 1e-9:
                out.append((inserted[1], material))
                placed = True
            out.append((t, m))
            z += t
        if not placed:
            out.append((inserted[1], material))
        self.layers = [(t, m) for t, m in out if t > 1e-9]
        return removed

    # -- 属性 -----------------------------------------------------------
    def weighted(self, depth0: float, depth1: float, water_table: float | None = None) -> dict[str, float]:
        """区间内按厚度加权的地层参数；给定地下水位时用浮重度。"""
        acc = {"gamma": 0.0, "c": 0.0, "phi": 0.0, "E": 0.0, "nu": 0.0, "k": 0.0}
        total = 0.0
        for a, b, m in self.boundaries():
            lo, hi = max(a, depth0), min(b, depth1)
            if hi <= lo:
                continue
            th = hi - lo
            gamma = m.gamma
            if water_table is not None and lo >= water_table:
                gamma = m.gamma_sub
            for key, val in (("gamma", gamma), ("c", m.c), ("phi", m.phi),
                             ("E", m.E), ("nu", m.nu), ("k", m.k)):
                acc[key] += val * th
            total += th
        if total <= 0:
            m = self.material_at(depth0)
            return {"gamma": m.gamma, "c": m.c, "phi": m.phi, "E": m.E,
                    "nu": m.nu, "k": m.k}
        return {k: v / total for k, v in acc.items()}

    def to_dict(self) -> dict:
        return {
            "top": round(self.top, 3),
            "layers": [
                {"thickness": round(t, 3), "material": m.key, "name": m.name}
                for t, m in self.layers
            ],
        }


def build_column(
    top: float,
    layer_specs: list[tuple[float, str]],
) -> LayeredColumn:
    """由 (厚度, 材料key) 序列构造分层柱。"""
    mats = []
    for t, key in layer_specs:
        mats.append((t, materials.get(key)))
    return LayeredColumn(top=top, layers=mats)


# ---------------------------------------------------------------------------
# 由钻孔重建整个区块的地质体
# ---------------------------------------------------------------------------
def reconstruct(
    boreholes: list[Borehole],
    xs: np.ndarray,
    ys: np.ndarray,
    surface: np.ndarray | None = None,
) -> tuple[list[LayeredColumn], dict]:
    """在网格上重建分层地质体。

    surface 给定时（真实工作流里的 DEM），地形直接取自地表高程，克里金只用于
    插值各层界的**埋深**——这比用稀疏钻孔外推绝对高程稳健得多。

    返回 (逐点分层柱列表, 诊断信息)。
    """
    if not boreholes:
        raise ValueError("至少需要一个钻孔")
    n_layers = max(len(b.layers) for b in boreholes)
    P = np.array([[b.x, b.y] for b in boreholes], dtype=float)
    T = np.column_stack([xs.ravel(), ys.ravel()])
    surf = (np.asarray(surface, dtype=float).ravel()
            if surface is not None else None)

    # 1) 各层顶面埋深的克里金插值
    surfaces: list[np.ndarray] = []
    for i in range(n_layers):
        vals, pts = [], []
        for bi, b in enumerate(boreholes):
            e = b.boundary_elevations()
            if i < len(e) - 1:
                vals.append(b.ground - e[i])
                pts.append(P[bi])
        if len(vals) < 2:
            surfaces.append(np.full(T.shape[0], np.nan))
            continue
        surfaces.append(krige(np.array(pts), np.array(vals), T))

    # 2) 岩性外推：取最近钻孔的岩性序列（同相带外推）
    #    真实项目中这一步换为指示克里金或序贯指示模拟；此处保持可解释的最小实现。
    dist = np.linalg.norm(T[:, None, :] - P[None, :, :], axis=2)   # (m, n)
    nearest = np.argmin(dist, axis=1)
    bore_sequences = [tuple(k for _, k in b.layers) for b in boreholes]

    # 3) 组装：埋深单调化 + 每点延伸到柱底
    cols: list[LayeredColumn] = []
    for j in range(T.shape[0]):
        depths = [0.0]
        for i in range(1, len(surfaces)):
            d = surfaces[i][j]
            if not np.isfinite(d):
                d = depths[-1] + 1.0
            depths.append(max(d, depths[-1] + 0.25))
        top = float(surf[j]) if surf is not None else float(surfaces[0][j] + 0.0)
        seq = bore_sequences[int(nearest[j])]
        specs = []
        for i in range(n_layers):
            bottom = depths[i + 1] if i + 1 < len(depths) else depths[i] + 2.0
            key = seq[min(i, len(seq) - 1)]
            specs.append((max(bottom - depths[i], 0.25), key))
        cols.append(build_column(top, specs))

    diag = {
        "boreholes": len(boreholes),
        "layers": n_layers,
        "cells": int(T.shape[0]),
        "surface_source": "DEM" if surf is not None else "kriged",
    }
    return cols, diag
