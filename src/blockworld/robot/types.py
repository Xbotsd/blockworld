"""机器人闭环里流动的数据类型。

设计约束：每一条测量都带**不确定度**和**来源**，每一次同化都留下先验/后验，
每一个决定都带**理由**与**置信度**。没有这两样，闭环就退化成"传感器覆盖模型"。
"""
from __future__ import annotations

from dataclasses import dataclass, field


# 测量种类 → 单位，集中声明，避免下游猜
KINDS: dict[str, str] = {
    "elevation": "m",       # 地表高程
    "water_table": "m",     # 地下水位埋深
    "layer_top": "m",       # 某层顶面埋深（material 指明是哪一层）
    "material_c": "kPa",    # 该深度处材料的黏聚力
    "material_phi": "deg",  # 该深度处材料的内摩擦角
}


@dataclass(frozen=True)
class Measurement:
    """一次扫描读数。"""

    kind: str
    x: float                      # 平面坐标 m
    y: float
    value: float
    sigma: float                  # 1σ 不确定度，单位同 KINDS[kind]
    device: str = "unknown"       # 产生它的终端/载荷
    depth: float = 0.0            # 取样深度 m（地表类测量为 0）
    material: str | None = None   # 目标材料 key（layer_top / material_* 用）
    t: float = 0.0                # 事件时间 s
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"未知测量种类 {self.kind!r}；可用：{', '.join(KINDS)}")
        if self.sigma <= 0:
            raise ValueError("测量不确定度必须为正")
        if self.depth < 0:
            raise ValueError("取样深度不能为负")

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "x": round(self.x, 3), "y": round(self.y, 3),
            "value": round(self.value, 4), "sigma": round(self.sigma, 4),
            "device": self.device, "depth": self.depth,
            "material": self.material, "t": round(self.t, 3),
            "unit": KINDS[self.kind], "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "Measurement":
        return Measurement(
            kind=d["kind"], x=float(d["x"]), y=float(d["y"]),
            value=float(d["value"]), sigma=float(d["sigma"]),
            device=d.get("device", "replay"), depth=float(d.get("depth", 0.0)),
            material=d.get("material"), t=float(d.get("t", 0.0)),
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class Assimilated:
    """一次测量对模型的改动（或拒绝）。"""

    measurement: Measurement
    accepted: bool
    prior: float
    posterior: float
    weight: float                 # 测量所占权重 0–1
    z_score: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "kind": self.measurement.kind,
            "tile": [round(self.measurement.x, 1), round(self.measurement.y, 1)],
            "prior": round(self.prior, 4),
            "posterior": round(self.posterior, 4),
            "weight": round(self.weight, 3),
            "z": round(self.z_score, 2),
            "device": self.measurement.device,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Decision:
    """一次自主判断。action ∈ continue / rescan / treat / certify / abort。"""

    action: str
    reason: str
    confidence: float             # 0–1
    payload: dict = field(default_factory=dict)
    tick: int = 0

    def to_dict(self) -> dict:
        return {"tick": self.tick, "action": self.action, "reason": self.reason,
                "confidence": round(self.confidence, 3), "payload": self.payload}


@dataclass
class SessionReport:
    """一次闭环运行的完整记录。"""

    ticks: list[dict]
    final: Decision | None
    coverage: float
    evidence: int
    measurements: int
    assimilated: int
    rejected: int
    before: dict
    after: dict
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ticks": self.ticks,
            "final": self.final.to_dict() if self.final else None,
            "coverage": round(self.coverage, 3),
            "evidence": self.evidence,
            "measurements": self.measurements,
            "assimilated": self.assimilated,
            "rejected": self.rejected,
            "before": self.before,
            "after": self.after,
            "meta": self.meta,
        }
