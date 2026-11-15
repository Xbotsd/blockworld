"""植物功能型（PFT）参数库 —— "生物群落替换" 的插槽。

对应分享对话里的机制：替换植被类型 = 换掉一整套生理参数（Vcmax、SLA、g1、
根系分布、耐阴/耐旱、气候包络），而不是改一个标签。区块最终显示什么生物群区，
由竞争与资源约束推演出来，而不是预设。
"""
from __future__ import annotations

from dataclasses import dataclass, replace


class EnvelopeError(ValueError):
    """PFT 被放到其气候包络之外。"""


@dataclass(frozen=True)
class PFT:
    key: str
    name: str
    biome: str            # 该 PFT 主导时区块呈现的生物群区名
    vcmax: float          # 最大羧化速率 μmol CO2 m⁻² s⁻¹
    sla: float            # 比叶面积 m²/kg C
    g1: float             # Medlyn 气孔导度斜率 kPa^0.5（越大越"费水"）
    root_depth: float     # 有效根深 m
    h_max: float          # 最大高度 m
    wood_density: float   # 木材密度 g/cm³（草本≈0.1）
    shade_tol: float      # 耐阴 0（阳性）–1（极耐阴）
    drought_tol: float    # 耐旱 0–1
    t_min: float          # 年均温下限 °C
    t_opt: float
    t_max: float
    p_min: float          # 年降水下限 mm
    p_max: float
    k_ext: float = 0.5    # 冠层消光系数
    mortality: float = 0.02   # 背景死亡率 /yr
    c_frac: float = 0.47      # 生物量含碳率
    is_woody: bool = True
    note: str = ""

    def suitability(self, temp_c: float, precip_mm: float) -> float:
        """气候包络隶属度 0–1：温度/降水的三角隶属取小。"""
        if temp_c <= self.t_min or temp_c >= self.t_max:
            return 0.0
        if temp_c <= self.t_opt:
            ft = (temp_c - self.t_min) / max(self.t_opt - self.t_min, 1e-6)
        else:
            ft = (self.t_max - temp_c) / max(self.t_max - self.t_opt, 1e-6)
        if precip_mm <= self.p_min or precip_mm >= self.p_max:
            fp = 0.0
        elif precip_mm <= 0.6 * (self.p_min + self.p_max):
            fp = (precip_mm - self.p_min) / max(
                0.6 * (self.p_min + self.p_max) - self.p_min, 1e-6
            )
        else:
            fp = (self.p_max - precip_mm) / max(
                self.p_max - 0.6 * (self.p_min + self.p_max), 1e-6
            )
        return max(0.0, min(1.0, min(ft, fp)))

    def check_envelope(self, temp_c: float, precip_mm: float, tol: float = 0.02) -> list[str]:
        s = self.suitability(temp_c, precip_mm)
        if s <= tol:
            return [
                f"{self.name} 的气候包络为 年均温 {self.t_min}–{self.t_max} °C、"
                f"年降水 {self.p_min:.0f}–{self.p_max:.0f} mm；"
                f"当前区块 {temp_c:.1f} °C / {precip_mm:.0f} mm 落在外推区"
            ]
        return []

    def with_params(self, **kw) -> "PFT":
        return replace(self, **kw)


def _p(**kw) -> PFT:
    return PFT(**kw)


LIBRARY: dict[str, PFT] = {
    p.key: p
    for p in [
        _p(key="temperate_broadleaf", name="温带落叶阔叶林", biome="温带阔叶林",
           vcmax=60, sla=20, g1=3.5, root_depth=1.5, h_max=28, wood_density=0.55,
           shade_tol=0.55, drought_tol=0.35, t_min=2, t_opt=13, t_max=23,
           p_min=620, p_max=1800, mortality=0.018),
        _p(key="boreal_conifer", name="寒温带针叶林", biome="北方针叶林",
           vcmax=45, sla=12, g1=2.5, root_depth=1.0, h_max=22, wood_density=0.45,
           shade_tol=0.75, drought_tol=0.45, t_min=-14, t_opt=4, t_max=14,
           p_min=320, p_max=1200, mortality=0.012),
        _p(key="mediterranean_shrub", name="地中海灌丛", biome="硬叶灌丛",
           vcmax=50, sla=14, g1=3.0, root_depth=2.2, h_max=4, wood_density=0.65,
           shade_tol=0.3, drought_tol=0.9, t_min=4, t_opt=16, t_max=26,
           p_min=280, p_max=780, mortality=0.03),
        _p(key="temperate_grass", name="温带草地（C3）", biome="温带草原",
           vcmax=55, sla=26, g1=4.5, root_depth=0.8, h_max=0.9, wood_density=0.10,
           shade_tol=0.05, drought_tol=0.55, t_min=-4, t_opt=14, t_max=26,
           p_min=260, p_max=1100, mortality=0.05, is_woody=False),
        _p(key="c4_grass", name="C4 草地", biome="C4 稀树草原",
           vcmax=75, sla=22, g1=2.0, root_depth=1.2, h_max=2.0, wood_density=0.12,
           shade_tol=0.05, drought_tol=0.8, t_min=8, t_opt=27, t_max=38,
           p_min=300, p_max=1200, mortality=0.05, is_woody=False,
           note="高光高温高水效：替换后蒸散下降、径流上升"),
        _p(key="tropical_rainforest", name="热带雨林", biome="热带雨林",
           vcmax=80, sla=22, g1=4.0, root_depth=2.5, h_max=45, wood_density=0.60,
           shade_tol=0.85, drought_tol=0.15, t_min=12, t_opt=26, t_max=34,
           p_min=1400, p_max=4000, mortality=0.015),
        _p(key="alpine_herb", name="高山草甸", biome="高山草甸",
           vcmax=35, sla=18, g1=3.2, root_depth=0.4, h_max=0.4, wood_density=0.10,
           shade_tol=0.0, drought_tol=0.30, t_min=-10, t_opt=5, t_max=12,
           p_min=200, p_max=900, mortality=0.06, is_woody=False),
    ]
}

BIOME_ORDER = [
    "tropical_rainforest",
    "temperate_broadleaf",
    "boreal_conifer",
    "mediterranean_shrub",
    "c4_grass",
    "temperate_grass",
    "alpine_herb",
]


def get(key: str) -> PFT:
    try:
        return LIBRARY[key]
    except KeyError:
        raise EnvelopeError(f"未知 PFT {key!r}；可用：{', '.join(sorted(LIBRARY))}") from None


def table() -> list[dict]:
    return [
        {
            "key": p.key,
            "name": p.name,
            "biome": p.biome,
            "vcmax": p.vcmax,
            "sla": p.sla,
            "g1": p.g1,
            "root_depth": p.root_depth,
            "h_max": p.h_max,
            "t_range": [p.t_min, p.t_max],
            "p_range": [p.p_min, p.p_max],
        }
        for p in LIBRARY.values()
    ]
