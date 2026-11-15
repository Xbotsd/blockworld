"""区块世界：地形、气候、地质柱、生物队列的容器。

一个区块（tile）是这套架构里唯一的最小共同单元：地质、水文、植被、动物都挂在
它上面，替换操作也以它为作用对象。世界本身是可快照/可回滚的，这样"替换 → 验证
→ 撤销"才能成为一次事务。
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from . import materials, pft as pft_mod


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class WorldConfig:
    nx: int = 48
    ny: int = 48
    cell: float = 100.0            # 区块边长 m
    seed: int = 20261115
    gravity: float = 9.81          # m/s²，可替换（月球 1.62 / 火星 3.71 / 小天体 ~1e-3）
    base_temp: float = 13.5        # 低海拔年均温 °C
    base_precip: float = 760.0     # 低海拔年降水 mm
    relief: float = 340.0          # 地形起伏 m
    river_width: float = 90.0      # 河谷半宽 m
    n_boreholes: int = 24
    sea_level: float = 0.0

    @property
    def width(self) -> float:
        return self.nx * self.cell

    @property
    def height(self) -> float:
        return self.ny * self.cell


# ---------------------------------------------------------------------------
# 生物队列
# ---------------------------------------------------------------------------
@dataclass
class Cohort:
    """一个植被队列：同一 PFT、相近尺寸的一群个体（FATES 的 cohort 简化版）。"""

    pft: str
    biomass: float        # kg C/m²（全器官合计）
    age: float            # 年
    height: float         # m

    def to_dict(self) -> dict:
        p = pft_mod.get(self.pft)
        return {
            "pft": self.pft,
            "name": p.name,
            "biomass": round(self.biomass, 3),
            "age": round(self.age, 1),
            "height": round(self.height, 2),
        }


# ---------------------------------------------------------------------------
# 区块
# ---------------------------------------------------------------------------
@dataclass
class Tile:
    ix: int
    iy: int
    x: float
    y: float
    elev: float
    col: geology.LayeredColumn
    temp: float = 12.0
    precip: float = 700.0
    wt: float = 3.0                       # 地下水位埋深 m
    discharge: float = 0.0                # 汇流后的流量 m³/s
    runoff: float = 0.0                   # 年径流深 mm
    infil: float = 0.0                    # 年入渗深 mm
    et: float = 0.0                       # 年蒸散深 mm
    lai: float = 0.0
    npp: float = 0.0                      # kg C/m²/yr
    cohort: list[Cohort] = field(default_factory=list)
    animals: list[int] = field(default_factory=list)
    patch_age: float = 60.0
    edited: bool = False
    edit_tags: list[str] = field(default_factory=list)

    # -- 便捷属性 -------------------------------------------------------
    @property
    def biomass(self) -> float:
        return sum(c.biomass for c in self.cohort)

    @property
    def dominant_pft(self) -> str | None:
        if not self.cohort:
            return None
        return max(self.cohort, key=lambda c: c.biomass).pft

    @property
    def biome(self) -> str:
        if self.biomass < 0.05:
            return "裸地/水体"
        return pft_mod.get(self.dominant_pft).biome


# ---------------------------------------------------------------------------
# 区域（替换/验证的作用范围）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Region:
    """矩形或圆形区块选区，坐标单位为米。"""

    kind: str = "rect"                 # rect | circle | all
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    r: float = 0.0

    @staticmethod
    def rect(x0: float, y0: float, x1: float, y1: float) -> "Region":
        return Region("rect", min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    @staticmethod
    def circle(cx: float, cy: float, r: float) -> "Region":
        return Region("circle", cx=cx, cy=cy, r=r)

    @staticmethod
    def whole() -> "Region":
        return Region("all")

    def contains(self, t: "Tile") -> bool:
        if self.kind == "all":
            return True
        if self.kind == "rect":
            return self.x0 <= t.x <= self.x1 and self.y0 <= t.y <= self.y1
        return (t.x - self.cx) ** 2 + (t.y - self.cy) ** 2 <= self.r**2

    def describe(self) -> str:
        if self.kind == "all":
            return "全域"
        if self.kind == "rect":
            return f"矩形[{self.x0:.0f},{self.y0:.0f}]–[{self.x1:.0f},{self.y1:.0f}]"
        return f"圆形({self.cx:.0f},{self.cy:.0f}) r={self.r:.0f}"

    def to_dict(self) -> dict:
        return {"kind": self.kind, "x0": self.x0, "y0": self.y0,
                "x1": self.x1, "y1": self.y1, "cx": self.cx, "cy": self.cy,
                "r": self.r}

    @staticmethod
    def from_dict(d: dict) -> "Region":
        kind = d.get("kind", "all")
        if kind == "all":
            return Region.whole()
        if kind == "rect":
            return Region.rect(float(d["x0"]), float(d["y0"]),
                               float(d["x1"]), float(d["y1"]))
        return Region.circle(float(d["cx"]), float(d["cy"]), float(d["r"]))


# ---------------------------------------------------------------------------
# 世界
# ---------------------------------------------------------------------------
class World:
    def __init__(self, config: WorldConfig, tiles: list[Tile], meta: dict | None = None):
        self.config = config
        self.tiles = tiles
        self.meta = meta or {}
        self._slope = np.zeros(len(tiles), dtype=float)
        self._fill_slopes()

    # -- 访问 -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.tiles)

    def idx(self, ix: int, iy: int) -> int:
        return iy * self.config.nx + ix

    def tile(self, ix: int, iy: int) -> Tile:
        return self.tiles[self.idx(ix, iy)]

    def at(self, x: float, y: float) -> Tile:
        ix = min(max(int(x / self.config.cell), 0), self.config.nx - 1)
        iy = min(max(int(y / self.config.cell), 0), self.config.ny - 1)
        return self.tile(ix, iy)

    def neighbours(self, ix: int, iy: int) -> list[Tile]:
        out = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            jx, jy = ix + dx, iy + dy
            if 0 <= jx < self.config.nx and 0 <= jy < self.config.ny:
                out.append(self.tile(jx, jy))
        return out

    def slope_deg(self, t: Tile) -> float:
        return float(self._slope[self.idx(t.ix, t.iy)])

    def region_tiles(self, region: "Region") -> list[Tile]:
        return [t for t in self.tiles if region.contains(t)]

    # -- 快照 / 回滚 -----------------------------------------------------
    def snapshot(self) -> "World":
        return copy.deepcopy(self)

    # -- 派生统计 -------------------------------------------------------
    def extent(self) -> dict:
        elev = np.array([t.elev for t in self.tiles])
        return {
            "min_elev": float(elev.min()),
            "max_elev": float(elev.max()),
            "mean_elev": float(elev.mean()),
            "width_m": self.config.width,
            "height_m": self.config.height,
            "cells": len(self.tiles),
        }

    def biome_map(self) -> list[str]:
        return [t.biome for t in self.tiles]

    def summary(self) -> dict:
        return {
            "config": {
                "nx": self.config.nx,
                "ny": self.config.ny,
                "cell_m": self.config.cell,
                "gravity": self.config.gravity,
                "seed": self.config.seed,
                "boreholes": self.config.n_boreholes,
            },
            "extent": self.extent(),
            **self.meta,
        }

    # -- 内部 -----------------------------------------------------------
    def _fill_slopes(self) -> None:
        nx, ny = self.config.nx, self.config.ny
        z = np.array([t.elev for t in self.tiles]).reshape(ny, nx)
        gy, gx = np.gradient(z, self.config.cell)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        self._slope = slope.ravel()


# ---------------------------------------------------------------------------
# 世界构建（向后兼容的入口）
# ---------------------------------------------------------------------------
def build(config: WorldConfig | None = None) -> World:
    """合成一个演示用世界：等价于 ``build_from(synthesize(config))``。

    库的正式入口是 :class:`blockworld.site.SiteSpec` + :func:`blockworld.site.build_from`；
    真实数据走那条路，这里只是示例与测试的快捷方式。
    """
    from .site import build_from, synthesize

    return build_from(synthesize(config))


def rebuild(world: World) -> World:
    """按当前配置重新生成（换种子）。"""
    cfg = copy.deepcopy(world.config)
    cfg.seed = world.config.seed + 1
    return build(cfg)


def __getattr__(name: str):
    """兼容旧的 ``from blockworld.world import truth_column``。"""
    if name == "truth_column":
        from .site import truth_column

        return truth_column
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
