"""事务式替换：成分 / 高程 / 水文 / 生物群落 / 重力。

每一次替换都是一次事务：
  1. 先做有效域检查（材料参数、本构、深度区间、气候包络）；
  2. 越界 → 直接拒绝并说明理由，不产生"数值可算、物理无意义"的世界；
  3. 通过 → 记录被替换的旧状态，支持 rollback，形成可追溯的编辑日志。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from . import materials, pft as pft_mod
from .world import Region, Tile, World


@dataclass
class EditRecord:
    index: int
    op: str
    region: str
    detail: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    tiles_affected: int = 0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "op": self.op,
            "region": self.region,
            "detail": self.detail,
            "warnings": self.warnings,
            "tiles_affected": self.tiles_affected,
        }

    def describe(self) -> str:
        bits = [f"#{self.index} {self.op} @ {self.region}"]
        for k, v in self.detail.items():
            bits.append(f"{k}={v}")
        return "，".join(bits)


class EditSession:
    """在一个世界上执行替换，并保留回滚与日志。"""

    def __init__(self, world: World):
        self.world = world
        self.records: list[EditRecord] = []
        self._undo: list[tuple[str, object]] = []

    # -- 基础设施 -------------------------------------------------------
    def _tiles(self, region: Region) -> list[Tile]:
        tiles = self.world.region_tiles(region)
        if not tiles:
            raise materials.DomainError(f"区域 {region.describe()} 内没有区块")
        return tiles

    def _commit(self, op: str, region: Region, detail: dict,
                warnings: list[str], tiles: list[Tile], undo) -> EditRecord:
        rec = EditRecord(len(self.records) + 1, op, region.describe(), detail,
                         warnings, len(tiles))
        self.records.append(rec)
        self._undo.append((op, undo))
        for t in tiles:
            t.edited = True
            t.edit_tags.append(op)
        return rec

    def rollback(self, n: int = 1) -> list[str]:
        """撤销最近 n 次替换。"""
        done = []
        for _ in range(min(n, len(self._undo))):
            op, undo = self._undo.pop()
            undo()
            rec = self.records.pop()
            done.append(f"撤销 {rec.describe()}")
        return done

    # -- 替换操作 -------------------------------------------------------
    def set_material(
        self,
        region: Region,
        material_key: str,
        *,
        top_depth: float = 0.0,
        bottom_depth: float | None = None,
        model: str | None = None,
    ) -> EditRecord:
        """把区域内 [top_depth, bottom_depth) 的地层替换为新材料。"""
        mat = materials.get(material_key)
        errors, warnings = mat.check(model)
        if errors:
            raise materials.DomainError("；".join(errors))

        tiles = self._tiles(region)
        sample = [t.col.material_at(top_depth) for t in tiles[:5]]
        for old in sample:
            warnings += materials.swap_check(old, mat)

        undo_layers = [(t, list(t.col.layers)) for t in tiles]
        removed_total = 0
        for t in tiles:
            bottom = bottom_depth if bottom_depth is not None else t.col.thickness
            bottom = min(bottom, t.col.thickness)
            removed = t.col.replace(top_depth, bottom, mat)
            removed_total += sum(r["thickness"] for r in removed)

        def undo():
            for tile, layers in undo_layers:
                tile.col.layers = layers

        return self._commit(
            "set_material", region,
            {"material": mat.name, "key": mat.key,
             "interval_m": f"{top_depth:.2f}–{'柱底' if bottom_depth is None else f'{bottom_depth:.2f}'}",
             "model": model or mat.models[0],
             "replaced_thickness_m": round(removed_total, 1)},
            warnings, tiles, undo,
        )

    def replace_component(
        self,
        region: Region,
        material_key: str,
        *,
        fraction: float = 1.0,
        model: str | None = None,
    ) -> EditRecord:
        """成分替换的另一种形式：只替换柱子里某一材料占的位置（按比例）。"""
        if not 0 < fraction <= 1.0:
            raise materials.DomainError("替换比例必须在 (0, 1] 内")
        target = materials.get(material_key)
        errors, warnings = target.check(model)
        if errors:
            raise materials.DomainError("；".join(errors))
        tiles = self._tiles(region)
        undo_cols = [(t, copy.deepcopy(t.col)) for t in tiles]
        n_changed = 0
        for t in tiles:
            bounds = t.col.boundaries()
            total = sum(b - a for a, b, m in bounds if m.key != target.key)
            quota = total * fraction
            new_layers = []
            for a, b, m in bounds:
                th = b - a
                if m.key == target.key:
                    new_layers.append((th, m))
                    continue
                take = min(th, quota)
                quota -= take
                if take > 0:
                    n_changed += 1
                    if th - take > 1e-6:
                        new_layers.append((th - take, m))
                    new_layers.append((take, target))
                else:
                    new_layers.append((th, m))
            t.col.layers = new_layers

        def undo():
            for tile, col in undo_cols:
                tile.col = col

        return self._commit("replace_component", region,
                            {"material": target.name, "fraction": fraction,
                             "slices": n_changed},
                            warnings, tiles, undo)

    def set_elevation(
        self, region: Region, *, delta: float | None = None, absolute: float | None = None
    ) -> EditRecord:
        """改变区块高程（填方/开挖/地形改造）。"""
        if delta is None and absolute is None:
            raise materials.DomainError("必须给出 delta 或 absolute")
        tiles = self._tiles(region)
        undo = [(t, t.elev, t.col.top) for t in tiles]
        for t in tiles:
            new = absolute if absolute is not None else t.elev + delta
            if new < 0.5:
                raise materials.DomainError(
                    f"区块 ({t.ix},{t.iy}) 目标高程 {new:.1f} m 低于可建模下限 0.5 m"
                )
            t.elev = new
            t.col.top = new

        def undo_fn():
            for tile, e, top in undo:
                tile.elev = e
                tile.col.top = top

        self.world._fill_slopes()
        return self._commit("set_elevation", region,
                            {"delta_m": delta, "absolute_m": absolute},
                            [], tiles, undo_fn)

    def set_water_table(self, region: Region, depth: float) -> EditRecord:
        """强制设定地下水位埋深（排水/灌溉/防渗工程）。"""
        if depth <= 0:
            raise materials.DomainError("地下水位埋深必须为正")
        tiles = self._tiles(region)
        undo = [(t, t.wt) for t in tiles]
        for t in tiles:
            t.wt = depth
            if depth < 0.5:
                t.wt = 0.5

        def undo_fn():
            for tile, wt in undo:
                tile.wt = wt

        return self._commit("set_water_table", region, {"depth_m": depth},
                            [], tiles, undo_fn)

    def set_precip(
        self, region: Region, *, factor: float | None = None, absolute: float | None = None
    ) -> EditRecord:
        """降水情景（气候情景或集水区调水）。"""
        tiles = self._tiles(region)
        undo = [(t, t.precip) for t in tiles]
        for t in tiles:
            new = absolute if absolute is not None else t.precip * (factor or 1.0)
            if new <= 0:
                raise materials.DomainError("降水必须为正")
            t.precip = new

        def undo_fn():
            for tile, p in undo:
                tile.precip = p

        return self._commit("set_precip", region,
                            {"factor": factor, "absolute_mm": absolute},
                            [], tiles, undo_fn)

    def set_pft(self, region: Region, pft_key: str, *, mode: str = "replace") -> EditRecord:
        """替换区块的植被成分。

        mode="replace"：清空原有队列，只保留该 PFT（检验极端替换）；
        mode="seed"：把它加入种子库，由竞争过程决定能否立足。
        """
        p = pft_mod.get(pft_key)
        tiles = self._tiles(region)
        outside = [t for t in tiles if p.suitability(t.temp, t.precip) <= 0.02]
        if len(outside) == len(tiles):
            t = tiles[0]
            raise pft_mod.EnvelopeError(
                f"{p.name} 的气候包络 {p.t_min}–{p.t_max} °C / {p.p_min:.0f}–{p.p_max:.0f} mm "
                f"与区块 {t.temp:.1f} °C / {t.precip:.0f} mm 不相容，替换被拒绝"
            )
        warnings = []
        if outside:
            warnings.append(
                f"{len(outside)}/{len(tiles)} 个区块处于该 PFT 包络之外，将依赖竞争淘汰"
            )
        undo = [(t, list(t.cohort), list(t.edit_tags)) for t in tiles]
        from .world import Cohort

        for t in tiles:
            if mode == "replace":
                t.cohort = []
            s = max(p.suitability(t.temp, t.precip), 0.05)
            existing = next((c for c in t.cohort if c.pft == p.key), None)
            if existing:
                existing.biomass = max(existing.biomass, 0.4 * s)
            else:
                t.cohort.append(Cohort(p.key, 0.5 * s, 0.0, p.h_max * 0.15))

        def undo_fn():
            for tile, cohorts, tags in undo:
                tile.cohort = cohorts
                tile.edit_tags = tags

        return self._commit("set_pft", region,
                            {"pft": p.name, "mode": mode}, warnings, tiles, undo_fn)

    def set_gravity(self, g: float) -> EditRecord:
        """替换局部重力（地球/月球/火星/小天体）。"""
        if not 0 < g <= 30:
            raise materials.DomainError(f"重力 {g} m/s² 超出可建模范围 (0, 30]")
        old = self.world.config.gravity
        self.world.config.gravity = g
        warnings = []
        if abs(g - 9.81) > 0.05:
            warnings.append(
                f"重力改为 {g:.2f} m/s²：承载力中的自重项按 γ·g/9.81 缩放，"
                f"但 c、φ 按原场地标定值保持不变（低重力下 c 主导，结论需重新解读）"
            )

        def undo_fn():
            self.world.config.gravity = old

        return self._commit("set_gravity", Region.whole(),
                            {"gravity": g, "earth_ratio": round(g / 9.81, 3)},
                            warnings, [], undo_fn)

    # -- 日志 -----------------------------------------------------------
    def log(self) -> list[dict]:
        return [r.to_dict() for r in self.records]
