"""可替换的"终端"（传感器载荷）。

`ScanDevice` 是唯一需要对接真实硬件的地方：给它一个世界（真值）和随机源，
它吐出一组带不确定度的 `Measurement`。接真实机器人时，只需实现同样的
`scan()`——从串口/ROS 话题/SDK 读取并转成 `Measurement`。

内置四个模拟终端，用于在没有硬件时把闭环跑通：

    lidar  高程扫描
    cpt    静力触探：把某深度的材料黏聚力/摩擦角读回来
    probe  地下水位探针
    replay 回放一段已录制的测量（等价于"换一台终端"）

注册表允许在不改任何调用点的情况下替换终端：

    register_device("lidar", MyRealLiDAR)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from ..world import Region, World
from .types import Measurement


class ScanDevice(Protocol):
    """一台扫描终端。"""

    name: str

    def scan(self, truth: World, rng: np.random.Generator) -> list[Measurement]:
        ...


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _tiles_in(truth: World, region: Region | None) -> list:
    tiles = truth.region_tiles(region) if region is not None else truth.tiles
    if not tiles:
        raise ValueError("扫描区域为空")
    return tiles


# ---------------------------------------------------------------------------
# 高程
# ---------------------------------------------------------------------------
@dataclass
class SimLiDAR:
    name: str = "lidar"
    per_scan: int = 6
    sigma: float = 0.05
    region: Region | None = None

    def scan(self, truth: World, rng: np.random.Generator) -> list[Measurement]:
        tiles = _tiles_in(truth, self.region)
        idx = rng.choice(len(tiles), size=min(self.per_scan, len(tiles)), replace=False)
        out = []
        for i in idx:
            t = tiles[int(i)]
            out.append(Measurement(
                kind="elevation", x=t.x, y=t.y,
                value=float(t.elev + rng.normal(0, self.sigma)),
                sigma=self.sigma, device=self.name,
            ))
        return out


# ---------------------------------------------------------------------------
# 静力触探：材料强度
# ---------------------------------------------------------------------------
@dataclass
class SimCPT:
    name: str = "cpt"
    per_scan: int = 2
    sigma_rel: float = 0.25
    phi_sigma: float = 1.5
    max_depth: float = 3.0
    region: Region | None = None

    def scan(self, truth: World, rng: np.random.Generator) -> list[Measurement]:
        tiles = _tiles_in(truth, self.region)
        out = []
        for _ in range(self.per_scan):
            t = tiles[int(rng.integers(0, len(tiles)))]
            depth = float(rng.uniform(0.5, self.max_depth))
            mat = t.col.material_at(depth)
            c_meas = max(0.01, mat.c * math.exp(rng.normal(0, self.sigma_rel)))
            out.append(Measurement(
                kind="material_c", x=t.x, y=t.y, value=c_meas,
                sigma=max(c_meas * self.sigma_rel, 0.05),
                device=self.name, depth=depth, material=mat.key,
            ))
            if rng.random() < 0.5:
                out.append(Measurement(
                    kind="material_phi", x=t.x, y=t.y,
                    value=float(mat.phi + rng.normal(0, self.phi_sigma)),
                    sigma=self.phi_sigma, device=self.name,
                    depth=depth, material=mat.key,
                ))
        return out


# ---------------------------------------------------------------------------
# 地下水位
# ---------------------------------------------------------------------------
@dataclass
class SimProbe:
    name: str = "probe"
    per_scan: int = 2
    sigma: float = 0.20
    region: Region | None = None

    def scan(self, truth: World, rng: np.random.Generator) -> list[Measurement]:
        tiles = _tiles_in(truth, self.region)
        out = []
        for _ in range(self.per_scan):
            t = tiles[int(rng.integers(0, len(tiles)))]
            out.append(Measurement(
                kind="water_table", x=t.x, y=t.y,
                value=max(0.1, float(t.wt + rng.normal(0, self.sigma))),
                sigma=self.sigma, device=self.name,
            ))
        return out


# ---------------------------------------------------------------------------
# 回放（换掉终端，只保留数据流）
# ---------------------------------------------------------------------------
@dataclass
class ReplayDevice:
    """把一串已录制的测量按批次吐出来：等价于接上一台"录好的终端"。"""

    measurements: tuple[Measurement, ...] = ()
    name: str = "replay"
    per_scan: int = 8
    _cursor: int = 0

    def scan(self, truth: World, rng: np.random.Generator) -> list[Measurement]:
        chunk = self.measurements[self._cursor:self._cursor + self.per_scan]
        self._cursor += len(chunk)
        return list(chunk)

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self.measurements)


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
DeviceFactory = Callable[..., ScanDevice]

DEVICES: dict[str, DeviceFactory] = {
    "lidar": SimLiDAR,
    "cpt": SimCPT,
    "probe": SimProbe,
}


def register_device(name: str, factory: DeviceFactory) -> None:
    """替换/新增终端。接真实硬件时在这里注册，而不是改 session。"""
    DEVICES[name] = factory


def make_device(name: str, **kwargs) -> ScanDevice:
    if name not in DEVICES:
        raise KeyError(f"未知终端 {name!r}；可用：{', '.join(sorted(DEVICES))}")
    return DEVICES[name](**kwargs)


def device_table() -> list[dict]:
    return [
        {"name": k, "class": v.__name__,
         "doc": (v.__doc__ or "").strip().splitlines()[0] if v.__doc__ else ""}
        for k, v in sorted(DEVICES.items())
    ]
