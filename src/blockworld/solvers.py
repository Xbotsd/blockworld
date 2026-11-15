"""求解器注册表：把"内置公式"换成"外部数值模型"的唯一接口。

库的用户通常有两类需求：
  1. 我只想换一条判据公式（例如用 PLAXIS 算出的 qu 代替 Vesic 解）；
  2. 我想整条目的验证都交给外部流程（跑完 FLAC3D 后回填一个 Verdict）。

对应两个扩展点：

    register_property("bearing_qu", fn)   # 只换公式
    register_criterion("landing", fn)     # 换整个验证器

两者都不改变其余代码路径：`Study`、敏感性、报告照旧工作。
"""
from __future__ import annotations

from typing import Any, Callable

# 属性级求解器：签名由名字约定，见 verifiers 中的调用点
PROPERTIES: dict[str, Callable[..., Any]] = {}

# 判据级求解器：fn(world, criteria) -> Verdict
CRITERIA: dict[str, Callable[..., Any]] = {}


def register_property(name: str, fn: Callable[..., Any]) -> None:
    """替换某个物理量的计算（例如 "bearing_qu"、"settlement_mm"）。"""
    PROPERTIES[name] = fn


def register_criterion(key: str, fn: Callable[..., Any]) -> None:
    """替换整个目的验证器（例如 "infrastructure"、"landing"）。"""
    CRITERIA[key] = fn


def clear(key: str | None = None) -> None:
    """撤销注册；不传 key 则清空全部（测试用）。"""
    if key is None:
        PROPERTIES.clear()
        CRITERIA.clear()
    else:
        PROPERTIES.pop(key, None)
        CRITERIA.pop(key, None)


def property_solver(name: str, default: Callable[..., Any]) -> Callable[..., Any]:
    return PROPERTIES.get(name, default)


def criterion_solver(key: str, default: Callable[..., Any]) -> Callable[..., Any]:
    return CRITERIA.get(key, default)


def normalize_capacity(result: Any) -> dict:
    """外部求解器可以返回 float 或 dict，统一成内部结构。"""
    if isinstance(result, dict):
        out = dict(result)
        out.setdefault("qu_kpa", float(out.get("qu_kpa", 0.0)))
        out.setdefault("terms", {})
        out["external"] = True
        return out
    return {"qu_kpa": float(result), "terms": {}, "external": True}


def active() -> dict:
    """当前注册情况（写进报告，便于追溯结论由谁算出）。"""
    return {"properties": sorted(PROPERTIES), "criteria": sorted(CRITERIA)}
