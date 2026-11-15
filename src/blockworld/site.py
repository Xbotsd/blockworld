"""场地描述与构建：把"数据"与"世界"分开。

这是把项目从"演示"变成"库"的关键一步：

    SiteSpec（纯数据，可存成 JSON/CSV）  --build_from()-->  World（可计算）

* 真实数据进入 SiteSpec：DEM 高程、钻孔、气温/降水场；
* `synthesize()` 只是 SiteSpec 的一个**显式**生成器，供示例与测试使用，
  它的产出会带 `provenance["source"]="synthetic"`，不会被误当成实测数据；
* 没有提供的数据不会被悄悄编造：缺 DEM / 缺钻孔直接报错。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import geology, pft as pft_mod
from .world import Cohort, Tile, World, WorldConfig


# ---------------------------------------------------------------------------
# 数据契约
# ---------------------------------------------------------------------------
@dataclass
class GridSpec:
    """规则网格：nx × ny 个区块，每块 cell 米，原点在 (x0, y0)。"""

    nx: int
    ny: int
    cell: float
    x0: float = 0.0
    y0: float = 0.0

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("网格尺寸必须为正")
        if self.cell <= 0:
            raise ValueError("区块边长必须为正")

    @property
    def width(self) -> float:
        return self.nx * self.cell

    @property
    def height(self) -> float:
        return self.ny * self.cell

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        X, Y = np.meshgrid(
            self.x0 + (np.arange(self.nx) + 0.5) * self.cell,
            self.y0 + (np.arange(self.ny) + 0.5) * self.cell,
        )
        return X, Y

    def to_dict(self) -> dict:
        return {"nx": self.nx, "ny": self.ny, "cell": self.cell,
                "x0": self.x0, "y0": self.y0}


@dataclass
class SiteSpec:
    """一个场地的全部输入数据。所有数组按 (ny, nx) 排列，行 = y 方向。"""

    grid: GridSpec
    elevation: np.ndarray | None = None          # m，DEM
    boreholes: tuple[geology.Borehole, ...] = ()
    temp: np.ndarray | None = None               # °C，年均
    precip: np.ndarray | None = None             # mm，年
    gravity: float = 9.81                        # m/s²
    seed: int = 0
    seed_pool: tuple[str, ...] | None = None     # 允许出现的 PFT；None = 按气候包络自动播种
    provenance: dict = field(default_factory=dict)
    truth: list[tuple[float, str]] | None = None  # 仅合成场地：逐区块真值剖面（用于校核）

    # -- 校验 -----------------------------------------------------------
    def validate(self) -> None:
        ny, nx = self.grid.ny, self.grid.nx
        if self.elevation is None:
            raise ValueError(
                "缺少 DEM 高程（site.elevation）。真实数据请提供 DEM；"
                "示例可用 synthesize() 生成。"
            )
        if self.elevation.shape != (ny, nx):
            raise ValueError(f"DEM 形状 {self.elevation.shape} 与网格 ({ny},{nx}) 不符")
        if not np.isfinite(self.elevation).all():
            raise ValueError("DEM 含 NaN/Inf")
        if not self.boreholes:
            raise ValueError("至少需要一个钻孔（site.boreholes）")
        bad = [(i, b.validate()) for i, b in enumerate(self.boreholes) if b.validate()]
        if bad:
            raise ValueError(f"钻孔 {bad[0][0]} 数据非法：{bad[0][1]}")
        for name in ("temp", "precip"):
            arr = getattr(self, name)
            if arr is not None and arr.shape != (ny, nx):
                raise ValueError(f"{name} 形状 {arr.shape} 与网格不符")
        if self.precip is not None and (self.precip <= 0).any():
            raise ValueError("降水必须为正")
        if not (0 < self.gravity <= 30):
            raise ValueError(f"重力 {self.gravity} 超出 (0, 30] m/s²")

    def climate_defaults(self) -> tuple[np.ndarray, np.ndarray]:
        """未提供气候场时的中性默认值（并在 provenance 里标记）。"""
        ny, nx = self.grid.ny, self.grid.nx
        temp = np.full((ny, nx), 12.0)
        precip = np.full((ny, nx), 800.0)
        self.provenance.setdefault("climate", "constant-default")
        return temp, precip

    def to_dict(self, *, inline_arrays: bool = False) -> dict:
        return {
            "grid": self.grid.to_dict(),
            "gravity": self.gravity,
            "seed": self.seed,
            "provenance": self.provenance,
            "seed_pool": list(self.seed_pool) if self.seed_pool else None,
            "boreholes": [
                {"x": b.x, "y": b.y, "ground": b.ground,
                 "layers": [[t, k] for t, k in b.layers], "source": b.source}
                for b in self.boreholes
            ],
            "elevation": self.elevation.tolist() if inline_arrays else None,
        }


# ---------------------------------------------------------------------------
# 构建：SiteSpec → World
# ---------------------------------------------------------------------------
def build_from(spec: SiteSpec) -> World:
    """用给出的场地数据构建世界（不做任何数据编造）。"""
    spec.validate()
    g = spec.grid
    X, Y = g.centres()
    elev = np.asarray(spec.elevation, dtype=float)
    temp, precip = (spec.temp, spec.precip)
    if temp is None or precip is None:
        t_d, p_d = spec.climate_defaults()
        temp = t_d if temp is None else temp
        precip = p_d if precip is None else precip

    cols, diag = geology.reconstruct(
        list(spec.boreholes), X.ravel(), Y.ravel(), surface=elev.ravel()
    )
    for c in cols:
        c.top = float(max(c.top, 1.0))

    tiles: list[Tile] = []
    for iy in range(g.ny):
        for ix in range(g.nx):
            j = iy * g.nx + ix
            tiles.append(
                Tile(ix=ix, iy=iy, x=float(X[iy, ix]), y=float(Y[iy, ix]),
                     elev=float(max(cols[j].top, 1.0)), col=cols[j],
                     temp=float(temp[iy, ix]), precip=float(precip[iy, ix]))
            )

    cfg = WorldConfig(nx=g.nx, ny=g.ny, cell=g.cell, gravity=spec.gravity,
                      seed=spec.seed, n_boreholes=len(spec.boreholes))
    world = World(cfg, tiles)
    world.meta["reconstruction"] = diag
    world.meta["provenance"] = dict(spec.provenance)
    world.meta["source"] = spec.provenance.get("source", "user-data")
    if spec.truth:
        err = [abs(spec.truth[j][0][0] - tiles[j].col.boundaries()[0][1])
               for j in range(len(tiles)) if spec.truth[j] and tiles[j].col.layers]
        world.meta["topsoil_depth_mae_m"] = round(float(np.mean(err)), 3)

    # 初始植被
    rng = np.random.default_rng(spec.seed + 991)
    pool = spec.seed_pool or tuple(p.key for p in pft_mod.LIBRARY.values())
    for t in tiles:
        for key in pool:
            p = pft_mod.get(key)
            s = p.suitability(t.temp, t.precip)
            if s > 0.05:
                t.cohort.append(
                    Cohort(key, 2.2 * s * (0.6 + 0.8 * rng.random()),
                           40 + rng.uniform(0, 40),
                           min(p.h_max * 0.7, p.h_max * (0.3 + 0.5 * s)))
                )
    world.meta["seed_pool"] = sorted({c.pft for t in tiles for c in t.cohort})
    return world


# ---------------------------------------------------------------------------
# 合成场地（显式：只用于示例/测试，provenance 会标注）
# ---------------------------------------------------------------------------
def _fractal(nx: int, ny: int, cell: float, rng: np.random.Generator, octaves: int = 5) -> np.ndarray:
    """多尺度正弦叠加的确定性分形纹理。返回大致 [-1, 1]。"""
    X, Y = np.meshgrid(np.arange(nx) * cell, np.arange(ny) * cell)
    out = np.zeros_like(X, dtype=float)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        fx = (o + 1) * 0.9 / (nx * cell) * 2 * math.pi
        fy = (o + 1) * 0.9 / (ny * cell) * 2 * math.pi
        ph1, ph2 = rng.uniform(0, 2 * math.pi, 2)
        out += amp * (
            0.6 * np.sin(fx * X + ph1) * np.cos(fy * Y + ph2)
            + 0.4 * np.sin(0.7 * fy * Y - 0.5 * fx * X + ph1 * 0.5)
        )
        total += amp
        amp *= 0.55
    return out / total


def _hill_field(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator,
                n_hills: int, width: float, height: float, relief: float) -> np.ndarray:
    """高斯山丘叠加：最大坡度 ≈ 0.86·amp/σ，比低频正弦更接近真实地形。"""
    out = np.zeros_like(X, dtype=float)
    for _ in range(n_hills):
        cx, cy = rng.uniform(0.05, 0.95) * width, rng.uniform(0.05, 0.95) * height
        sx = rng.uniform(0.05, 0.13) * width
        sy = rng.uniform(0.05, 0.13) * height
        amp = rng.uniform(0.35, 1.0) * relief
        out += amp * np.exp(-(((X - cx) / sx) ** 2 + ((Y - cy) / sy) ** 2))
    return out


def _river_center(y: float, ny: int, cell: float, phase: float) -> float:
    span = ny * cell
    return 0.5 * span + 0.18 * span * math.sin(2 * math.pi * y / (0.9 * span) + phase)


def truth_column(elev: float, valley_dist: float, relief: float,
                 wetness: float) -> list[tuple[float, str]]:
    """合成场地的"真值"剖面（仅生成器使用，模型看不到它）。"""
    band = float(np.clip((elev - 12.0) / max(relief, 1.0), 0, 1))
    if valley_dist < 110:
        return [(0.8, "silt"), (2.2, "clay_soft"), (3.5, "sand_dense"),
                (4.0 + 3 * wetness, "gravel_dense"), (8.0, "weathered_rock")]
    if valley_dist < 420:
        return [(0.6, "sand_dense"), (2.5 + 2 * band, "silt"),
                (4.5, "weathered_rock"), (8.0, "sound_rock")]
    if band > 0.5:
        return [(0.4, "weathered_rock"), (6.0 + 4 * band, "sound_rock")]
    return [(0.8, "sand_dense"), (3.0, "weathered_rock"), (7.0, "sound_rock")]


def synthesize(cfg: WorldConfig | None = None) -> SiteSpec:
    """生成一个合成场地（DEM + 钻孔 + 气候场）。仅用于示例与测试。"""
    cfg = cfg or WorldConfig()
    rng = np.random.default_rng(cfg.seed)
    nx, ny, cell = cfg.nx, cfg.ny, cfg.cell
    W, H = nx * cell, ny * cell
    X, Y = np.meshgrid(np.arange(nx) * cell + cell / 2,
                       np.arange(ny) * cell + cell / 2)

    # 地形
    hills = _hill_field(X, Y, rng, n_hills=7, width=W, height=H, relief=cfg.relief)
    hills = hills / max(hills.max(), 1e-9) * cfg.relief
    texture = _fractal(nx, ny, cell, rng, octaves=6) * 5.0
    elev = 14.0 + hills + texture
    phase = rng.uniform(0, 2 * math.pi)
    center = np.array([_river_center(y, ny, cell, phase)
                       for y in np.arange(ny) * cell + cell / 2])
    dxr = np.abs(X - center[:, None])
    valley = np.exp(-(dxr / cfg.river_width) ** 2)
    elev = elev - 0.12 * cfg.relief * valley
    elev = elev - elev.min() + 6.0

    # 气候
    lat_grad = 9.0 * (0.5 - Y / H)
    temp = cfg.base_temp - 0.0065 * (elev - elev.min()) + lat_grad
    orographic = 0.85 * (elev - elev.min())
    crest = 0.62 * W
    lee = np.where(X > crest, np.exp(-((X - crest) / (0.22 * W)) ** 2), 0.0)
    precip = np.clip(cfg.base_precip + orographic - 320.0 * lee + 90.0 * valley,
                     260.0, 2600.0)

    # 真值 + 钻孔
    truth = [truth_column(float(elev[iy, ix]), float(dxr[iy, ix]), cfg.relief,
                          float(np.clip(valley[iy, ix], 0, 1)))
             for iy in range(ny) for ix in range(nx)]
    n_bh = cfg.n_boreholes
    n_ctrl = min(6, n_bh)
    pts = np.column_stack([rng.uniform(0, nx * cell, n_bh),
                           rng.uniform(0, ny * cell, n_bh)])
    pts[:n_ctrl, 0] = center[[int(o * ny / n_ctrl) for o in range(n_ctrl)]]
    pts[:n_ctrl, 1] = np.arange(n_ctrl) * (ny * cell / n_ctrl) + cell / 2
    boreholes = []
    for k, (bx, by) in enumerate(pts):
        ix = min(int(bx / cell), nx - 1)
        iy = min(int(by / cell), ny - 1)
        layers = []
        for t, m in truth[iy * nx + ix]:
            noise = 0.0 if k < n_ctrl else rng.normal(0, 0.12)
            layers.append((max(0.3, t * (1 + noise)), m))
        boreholes.append(
            geology.Borehole(bx, by, float(elev[iy, ix] + rng.normal(0, 0.25)),
                             tuple(layers), source="control" if k < n_ctrl else "survey")
        )

    dem = elev + rng.normal(0, 0.25, size=elev.shape)
    return SiteSpec(
        grid=GridSpec(nx, ny, cell),
        elevation=dem,
        boreholes=tuple(boreholes),
        temp=temp,
        precip=precip,
        gravity=cfg.gravity,
        seed=cfg.seed,
        truth=truth,
        provenance={
            "source": "synthetic",
            "generator": "blockworld.site.synthesize",
            "note": "真值剖面来自同一生成规则，重建精度指标仅用于自洽性检查，"
                    "不能当作实地精度。",
        },
    )
