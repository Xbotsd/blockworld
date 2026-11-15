"""材料库与本构有效域（第一层：可替换的物性对象）。

设计原则（对应分享对话里的"可替换性必须与本构适用域绑定"）：
  1. 一次成分替换 = 材料本体 + 本构模型 + 全套参数 的原子切换；
  2. 每个材料携带自己的有效域，越界替换直接被拒绝，而不是"数值能算、物理无意义"；
  3. 同一材料在不同重力/围压下不自动外推——由 verifier 负责说明适用边界。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

# ---------------------------------------------------------------------------
# 本构模型登记表
# ---------------------------------------------------------------------------
CONSTITUTIVE_MODELS: dict[str, str] = {
    "mohr-coulomb": "Mohr–Coulomb 弹塑性（通用，强度控制）",
    "hardening-soil": "Hardening Soil（刚度分层，变形控制）",
    "drucker-prager": "修正 Drucker–Prager（数值高效，大变形/冲击）",
    "undrained": "不排水总应力（φu≈0，软黏土短期行为）",
}


class DomainError(ValueError):
    """替换越出物理/本构有效域。"""


@dataclass(frozen=True)
class Material:
    """一种可替换的地质/工程材料。单位：kN, m, kPa, MPa, m/s。"""

    key: str
    name: str
    gamma: float          # 天然重度 kN/m³
    c: float              # 有效黏聚力 kPa
    phi: float            # 内摩擦角 °
    E: float              # 变形模量 MPa
    nu: float             # 泊松比
    k: float              # 渗透系数 m/s
    models: tuple[str, ...] = ("mohr-coulomb",)
    phi_max: float = 45.0
    c_max: float = 500.0
    gamma_min: float = 5.0
    gamma_max: float = 30.0
    source: str = ""
    note: str = ""

    # -- 派生量 ---------------------------------------------------------
    @property
    def gamma_sub(self) -> float:
        """水下有效重度（浮重度），γ' = γ - γw。"""
        return max(self.gamma - 9.81, 0.5)

    def with_params(self, **kw) -> "Material":
        """返回替换了部分参数的副本（用于敏感性扫描）。"""
        return replace(self, **kw)

    # -- 有效域检查 -----------------------------------------------------
    def check(self, model: str | None = None) -> tuple[list[str], list[str]]:
        """返回 (errors, warnings)。errors 非空则替换必须被拒绝。"""
        errors: list[str] = []
        warnings: list[str] = []
        if not (0.0 <= self.phi < self.phi_max):
            errors.append(f"内摩擦角 φ={self.phi:.1f}° 越界 [0, {self.phi_max})")
        if not (0.0 <= self.c <= self.c_max):
            errors.append(f"黏聚力 c={self.c:.1f} kPa 越界 [0, {self.c_max}]")
        if not (self.gamma_min <= self.gamma <= self.gamma_max):
            errors.append(
                f"重度 γ={self.gamma:.1f} kN/m³ 越界 [{self.gamma_min}, {self.gamma_max}]"
            )
        if self.E <= 0:
            errors.append(f"变形模量 E={self.E:.1f} MPa 必须为正")
        if not (0.0 <= self.nu < 0.5):
            errors.append(f"泊松比 ν={self.nu:.2f} 越界 [0, 0.5)")
        if self.k <= 0:
            errors.append(f"渗透系数 k={self.k:g} m/s 必须为正")
        if model is not None and model not in self.models:
            errors.append(
                f"材料 {self.name} 不支持本构 {model}；可用：{', '.join(self.models)}"
            )
        # 物理上"能算但无意义"的组合
        if model == "undrained" and self.phi > 5:
            errors.append("不排水总应力模型要求 φu≈0，与当前摩擦角冲突")
        if self.c == 0 and self.phi < 20:
            warnings.append("无黏聚力且低摩擦角：几乎不具抗剪能力，稳定性判据会必然失败")
        if self.c > 0 and self.phi > 40 and self.E < 50:
            warnings.append("高强度低刚度组合（似胶结/风化岩）：变形可能先于强度失效")
        return errors, warnings

    def validate(self, model: str | None = None) -> None:
        errors, _ = self.check(model)
        if errors:
            raise DomainError("；".join(errors))


def _m(**kw) -> Material:
    return Material(**kw)


# ---------------------------------------------------------------------------
# 材料库：地球常规土 + 行星风化层（用于"着陆可行性"目的）
# ---------------------------------------------------------------------------
LIBRARY: dict[str, Material] = {
    m.key: m
    for m in [
        _m(key="gravel_dense", name="密实砂砾", gamma=21.0, c=0.0, phi=40.0,
           E=80.0, nu=0.25, k=1e-3, models=("mohr-coulomb", "hardening-soil"),
           source="PLAXIS 材料适用性对照（砾石）"),
        _m(key="sand_dense", name="密实砂", gamma=19.5, c=2.0, phi=36.0,
           E=45.0, nu=0.28, k=1e-4, models=("mohr-coulomb", "hardening-soil")),
        _m(key="sand_loose", name="松散砂", gamma=17.5, c=0.0, phi=29.0,
           E=18.0, nu=0.30, k=5e-4, models=("mohr-coulomb",),
           note="易液化；饱和时需按不排水强度折减"),
        _m(key="silt", name="粉土", gamma=18.4, c=6.0, phi=26.0,
           E=12.0, nu=0.32, k=1e-6, models=("mohr-coulomb", "hardening-soil")),
        _m(key="clay_soft", name="软黏土", gamma=17.0, c=12.0, phi=14.0,
           E=4.0, nu=0.35, k=1e-9,
           models=("mohr-coulomb", "undrained", "hardening-soil", "drucker-prager"),
           note="不排水强度 su≈12 kPa，压缩性高"),
        _m(key="clay_stiff", name="硬黏土", gamma=19.0, c=45.0, phi=22.0,
           E=25.0, nu=0.33, k=1e-10,
           models=("mohr-coulomb", "undrained", "hardening-soil")),
        _m(key="peat", name="泥炭", gamma=10.5, c=8.0, phi=18.0,
           E=1.2, nu=0.40, k=1e-7, phi_max=30.0,
           models=("mohr-coulomb", "undrained"),
           source="PLAXIS 对照表（泥炭）", note="极低刚度，沉降控制"),
        _m(key="weathered_rock", name="强风化岩", gamma=22.0, c=40.0, phi=32.0,
           E=150.0, nu=0.24, k=1e-7, phi_max=45.0,
           models=("mohr-coulomb", "hardening-soil", "drucker-prager")),
        _m(key="sound_rock", name="中风化岩", gamma=25.0, c=250.0, phi=42.0,
           E=4000.0, nu=0.20, k=1e-9, c_max=2000.0,
           models=("mohr-coulomb", "drucker-prager")),
        # --- 行星 / 小天体风化层（重力由 World.gravity 显式注入）---
        _m(key="lunar_regolith", name="月球风化层", gamma=15.0, c=0.5, phi=35.0,
           E=8.0, nu=0.30, k=1e-8, gamma_min=8.0, gamma_max=22.0,
           models=("mohr-coulomb", "drucker-prager"),
           source="ESA 着陆动力学名义风化层（承载力≈1.3 N/cm²，地基反力模量≈0.1 N/cm²/mm）",
           note="低重力下 c 主导承载排序"),
        _m(key="mars_regolith", name="火星风化层", gamma=14.5, c=1.2, phi=34.0,
           E=10.0, nu=0.30, k=1e-8, gamma_min=8.0, gamma_max=22.0,
           models=("mohr-coulomb", "drucker-prager")),
        _m(key="asteroid_regolith", name="小天体风化层", gamma=9.0, c=0.05, phi=30.0,
           E=0.5, nu=0.35, k=1e-9, gamma_min=3.0, gamma_max=20.0,
           models=("mohr-coulomb", "drucker-prager"),
           note="微重力：黏聚力与摩擦角主导，密度/重力项权重极低"),
        _m(key="cemented_regolith", name="胶结风化层", gamma=16.5, c=25.0, phi=38.0,
           E=60.0, nu=0.28, k=1e-8, gamma_min=8.0, gamma_max=24.0,
           models=("mohr-coulomb", "drucker-prager"),
           note="场地改良后的目标材料"),
    ]
}

# 材料族：同族互换不改变主导失效模式，跨族互换需要重新标定判据阈值
FAMILIES: dict[str, tuple[str, ...]] = {
    "cohesionless": ("gravel_dense", "sand_dense", "sand_loose", "asteroid_regolith"),
    "cohesive": ("clay_soft", "clay_stiff", "silt", "peat"),
    "rock": ("weathered_rock", "sound_rock"),
    "regolith": ("lunar_regolith", "mars_regolith", "cemented_regolith"),
}


def get(key: str) -> Material:
    try:
        return LIBRARY[key]
    except KeyError:
        raise DomainError(f"未知材料 {key!r}；可用：{', '.join(sorted(LIBRARY))}") from None


def family(key: str) -> str | None:
    for fam, keys in FAMILIES.items():
        if key in keys:
            return fam
    return None


def swap_check(old: Material, new: Material) -> list[str]:
    """族间替换提示：跨族替换会改变主导失效模式。"""
    fo, fn = family(old.key), family(new.key)
    if fo and fn and fo != fn:
        return [
            f"跨族替换 {old.name} → {new.name}：主导失效模式由 {fo} 变为 {fn}，"
            f"既有判据阈值（安全系数、允许沉降）需重新标定"
        ]
    return []


def table(materials: Iterable[Material] | None = None) -> list[dict]:
    items = list(materials or LIBRARY.values())
    return [
        {
            "key": m.key,
            "name": m.name,
            "gamma": m.gamma,
            "c": m.c,
            "phi": m.phi,
            "E": m.E,
            "nu": m.nu,
            "k": m.k,
            "models": list(m.models),
            "family": family(m.key) or "-",
        }
        for m in items
    ]
