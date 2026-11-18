"""同化：把实测数据折进模型，并留下先验/后验与拒绝理由。

两条规则：

1. **逆方差融合**：测量权重 `w = σ_prior² / (σ_prior² + σ_meas²)`；
   空间上按距离衰减，一次点测影响邻域而不是整个世界。
2. **异常门控**：`|测量 − 先验| / √(σ_prior² + σ_meas²) > z_gate` 直接拒绝。
   传感器故障、标定漂移、贴错坐标，都会在这里被挡下，而不是污染模型。

拒绝不抛异常——闭环要能继续跑，但每一次拒绝都记录在审计流里。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import materials
from ..world import Tile, World
from .types import Assimilated, Measurement

# 先验不确定度：模型本身的误差量级（不是测量误差）
PRIOR_SIGMA: dict[str, float] = {
    "elevation": 0.50,       # m，DEM/重建误差
    "water_table": 1.00,     # m
    "layer_top": 0.50,       # m
    "material_c": 0.40,      # 相对值（40%）
    "material_phi": 4.0,     # 度
}


@dataclass
class Assimilator:
    world: World
    gate_sigma: float = 4.0
    influence_m: float = 150.0     # 点测的平面影响半径
    prior_sigma: dict[str, float] = None

    def __post_init__(self) -> None:
        self.prior_sigma = dict(PRIOR_SIGMA if self.prior_sigma is None else self.prior_sigma)
        self.audit: list[Assimilated] = []
        self.accepted = 0
        self.rejected = 0

    # -- 入口 -----------------------------------------------------------
    def apply(self, m: Measurement) -> Assimilated:
        handler = {
            "elevation": self._elevation,
            "water_table": self._water_table,
            "layer_top": self._layer_top,
            "material_c": self._material_prop,
            "material_phi": self._material_prop,
        }[m.kind]
        rec = handler(m)
        self.audit.append(rec)
        if rec.accepted:
            self.accepted += 1
        else:
            self.rejected += 1
        return rec

    def apply_all(self, measurements: list[Measurement]) -> list[Assimilated]:
        out = [self.apply(m) for m in measurements]
        self.world._fill_slopes()          # 高程变了，坡度场要重算
        return out

    # -- 内部 -----------------------------------------------------------
    def _target_tile(self, m: Measurement) -> Tile | None:
        for t in self.world.tiles:
            if abs(t.x - m.x) <= self.world.config.cell / 2 and \
               abs(t.y - m.y) <= self.world.config.cell / 2:
                return t
        return None

    def _weight(self, prior_sigma: float, meas_sigma: float, dist: float) -> float:
        w = prior_sigma**2 / (prior_sigma**2 + meas_sigma**2)
        return w * math.exp(-((dist / max(self.influence_m, 1e-6)) ** 2))

    def _gate(self, prior: float, value: float, prior_sigma: float,
              meas_sigma: float) -> tuple[bool, float]:
        z = abs(value - prior) / math.sqrt(prior_sigma**2 + meas_sigma**2 + 1e-12)
        return z <= self.gate_sigma, z

    # -- 各类测量 -------------------------------------------------------
    def _elevation(self, m: Measurement) -> Assimilated:
        t = self._target_tile(m)
        if t is None:
            return Assimilated(m, False, float("nan"), float("nan"), 0.0,
                               reason="落点不在任何区块内")
        prior = t.elev
        ps = self.prior_sigma["elevation"]
        ok, z = self._gate(prior, m.value, ps, m.sigma)
        if not ok:
            return Assimilated(m, False, prior, prior, 0.0, z, "超出异常门控（z>4）")
        w = self._weight(ps, m.sigma, 0.0)
        post = prior + w * (m.value - prior)
        # 一个激光点只说明它所在区块的高度；不对邻域做平滑（会把坡面抹平）
        t.elev = post
        t.col.top = post
        return Assimilated(m, True, prior, post, w, z, "单点高程融合")

    def _water_table(self, m: Measurement) -> Assimilated:
        t = self._target_tile(m)
        if t is None:
            return Assimilated(m, False, float("nan"), float("nan"), 0.0,
                               reason="落点不在任何区块内")
        prior = t.wt
        ps = self.prior_sigma["water_table"]
        ok, z = self._gate(prior, m.value, ps, m.sigma)
        if not ok:
            return Assimilated(m, False, prior, prior, 0.0, z, "超出异常门控（z>4）")
        w = self._weight(ps, m.sigma, 0.0)
        post = float(np.clip(prior + w * (m.value - prior), 0.1, 30.0))
        t.wt = post
        return Assimilated(m, True, prior, post, w, z, "单点水位融合")

    def _layer_top(self, m: Measurement) -> Assimilated:
        t = self._target_tile(m)
        if t is None:
            return Assimilated(m, False, float("nan"), float("nan"), 0.0,
                               reason="落点不在任何区块内")
        if len(t.col.layers) < 2:
            return Assimilated(m, False, float("nan"), float("nan"), 0.0,
                               reason="柱内只有一层，无法调整层界")
        prior = t.col.layers[0][0]
        ps = self.prior_sigma["layer_top"]
        ok, z = self._gate(prior, m.value, ps, m.sigma)
        if not ok:
            return Assimilated(m, False, prior, prior, 0.0, z, "超出异常门控（z>4）")
        w = self._weight(ps, m.sigma, 0.0)
        post = prior + w * (m.value - prior)
        if not 0.05 <= post <= prior + t.col.layers[1][0] - 0.05:
            return Assimilated(m, False, prior, prior, 0.0, z,
                               f"后验层厚 {post:.2f} m 会抹掉相邻层，拒绝")
        thick0, mat0 = t.col.layers[0]
        thick1, mat1 = t.col.layers[1]
        delta = post - thick0
        t.col.layers[0] = (post, mat0)
        t.col.layers[1] = (thick1 - delta, mat1)
        return Assimilated(m, True, prior, post, w, z, "层界融合")

    def _material_prop(self, m: Measurement) -> Assimilated:
        key = m.material
        if not key:
            return Assimilated(m, False, float("nan"), float("nan"), 0.0,
                               reason="测量未指明材料，无法定位")
        prop = "c" if m.kind == "material_c" else "phi"
        t = self._target_tile(m)
        if t is None:
            return Assimilated(m, False, float("nan"), float("nan"), 0.0,
                               reason="落点不在任何区块内")

        # 1) 柱内没有该材料 → 扫描给出了"这里是什么"，按深度重分类
        reclassified = False
        if not any(mm.key == key for _, mm in t.col.layers):
            idx = t.col.layer_index_at(m.depth)
            th, old = t.col.layers[idx]
            new_mat = materials.get(key)
            errs, _ = new_mat.check()
            if errs:
                return Assimilated(m, False, float("nan"), float("nan"), 0.0,
                                   reason=f"重分类目标 {key} 越出有效域：{errs[0]}")
            t.col.layers[idx] = (th, new_mat)
            reclassified = True

        layer = next((mm for _, mm in t.col.layers if mm.key == key), None)
        if layer is None:
            return Assimilated(m, False, float("nan"), float("nan"), 0.0,
                               reason=f"该区块柱内没有材料 {key}")
        prior = float(getattr(layer, prop))
        ps = self.prior_sigma[f"material_{prop}"]
        ps = ps * prior if prop == "c" else ps       # c 用相对先验
        ok, z = self._gate(prior, m.value, max(ps, 1e-6), m.sigma)
        if not ok:
            return Assimilated(m, False, prior, prior, 0.0, z, "超出异常门控（z>4）")

        w = self._weight(max(ps, 1e-6), m.sigma, 0.0)
        post = prior + w * (m.value - prior)
        changed = 0
        for nt in self.world.tiles:
            d = math.hypot(nt.x - m.x, nt.y - m.y)
            if d > self.influence_m:
                continue
            w2 = self._weight(max(ps, 1e-6), m.sigma, d)
            if w2 <= 1e-6:
                continue
            new_layers = []
            touched = False
            for th, mm in nt.col.layers:
                if mm.key != key:
                    new_layers.append((th, mm))
                    continue
                val = float(getattr(mm, prop)) + w2 * (m.value - float(getattr(mm, prop)))
                cand = mm.with_params(**{prop: val})
                errs, _ = cand.check()
                if errs:
                    new_layers.append((th, mm))       # 越界就不动，保留原值
                    continue
                new_layers.append((th, cand))
                touched = True
            if touched:
                nt.col.layers = new_layers
                changed += 1
        if changed == 0 and not reclassified:
            return Assimilated(m, False, prior, prior, 0.0, z,
                               "后验参数越出材料有效域，全部拒绝")
        note = f"更新 {changed} 个区块的 {key}.{prop}"
        if reclassified:
            note += "（含按扫描重新分类）"
        return Assimilated(m, True, prior, post, w, z, note)
