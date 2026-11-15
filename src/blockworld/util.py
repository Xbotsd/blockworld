"""小工具：把 numpy 标量/数组转成原生类型，保证 JSON 可序列化。"""
from __future__ import annotations

from typing import Any


def native(value: Any) -> Any:
    """单个值的类型归一化（numpy 标量 → 原生标量）。"""
    if type(value) in (bool, int, float, str) or value is None:
        return value
    if hasattr(value, "item"):          # numpy 标量（np.float64 也是 float 子类）
        return value.item()
    if hasattr(value, "tolist"):        # numpy 数组
        return value.tolist()
    return value


def jsonable(obj: Any) -> Any:
    """递归归一化，供 json.dumps 使用。"""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    if hasattr(obj, "item"):
        return jsonable(obj.item())
    if hasattr(obj, "tolist"):
        return jsonable(obj.tolist())
    return str(obj)
