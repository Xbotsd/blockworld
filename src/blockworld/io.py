"""数据读写：把场地数据从文件读进来，或把结果写成文件。

支持的最小格式（够用、可手写、可被任何工具生成）：

    site.json
    {
      "grid": {"nx": 24, "ny": 24, "cell": 100, "x0": 0, "y0": 0},
      "gravity": 9.81,
      "seed": 1,
      "elevation": {"csv": "dem.csv"},          // 或 [[...], [...]]
      "temp":      {"csv": "temp.csv"},         // 可选
      "precip":    {"csv": "precip.csv"},       // 可选
      "boreholes": {"csv": "boreholes.csv"},    // 或 [{"x":..,"y":..,"ground":..,"layers":[[厚度,"材料"],..]}]
      "provenance": {"source": "实测/公开数据集/..."}
    }

    dem.csv / temp.csv / precip.csv   纯数值矩阵，ny 行 × nx 列
    boreholes.csv                     表头 x,y,ground,top,bottom,material[,source]，每行一层
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from . import materials
from .geology import Borehole
from .site import GridSpec, SiteSpec


class SiteFormatError(ValueError):
    """输入文件格式不对。"""


# ---------------------------------------------------------------------------
# 读
# ---------------------------------------------------------------------------
def load_site(path: str | Path) -> SiteSpec:
    path = Path(path)
    if not path.exists():
        raise SiteFormatError(f"找不到场地文件：{path}")
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    base = path.parent

    grid_doc = doc.get("grid")
    if not grid_doc:
        raise SiteFormatError("site.json 缺少 grid 段")
    grid = GridSpec(nx=int(grid_doc["nx"]), ny=int(grid_doc["ny"]),
                    cell=float(grid_doc["cell"]),
                    x0=float(grid_doc.get("x0", 0.0)),
                    y0=float(grid_doc.get("y0", 0.0)))

    elevation = _load_field(doc.get("elevation"), base, grid, "elevation")
    temp = _load_field(doc.get("temp"), base, grid, "temp")
    precip = _load_field(doc.get("precip"), base, grid, "precip")

    bore_doc = doc.get("boreholes")
    if bore_doc is None:
        raise SiteFormatError("site.json 缺少 boreholes")
    if isinstance(bore_doc, dict) and "csv" in bore_doc:
        boreholes = load_boreholes_csv(base / bore_doc["csv"])
    elif isinstance(bore_doc, list):
        boreholes = tuple(_borehole_from_dict(b) for b in bore_doc)
    else:
        raise SiteFormatError("boreholes 需要是 {\"csv\": ...} 或列表")

    seed_pool = doc.get("seed_pool")
    return SiteSpec(
        grid=grid,
        elevation=elevation,
        boreholes=tuple(boreholes),
        temp=temp,
        precip=precip,
        gravity=float(doc.get("gravity", 9.81)),
        seed=int(doc.get("seed", 0)),
        seed_pool=tuple(seed_pool) if seed_pool else None,
        provenance=dict(doc.get("provenance") or {}),
    )


def _load_field(doc, base: Path, grid: GridSpec, name: str) -> np.ndarray | None:
    if doc is None:
        return None
    if isinstance(doc, dict) and "csv" in doc:
        arr = load_grid_csv(base / doc["csv"])
    elif isinstance(doc, list):
        arr = np.asarray(doc, dtype=float)
    else:
        raise SiteFormatError(f"{name} 需要是 {{\"csv\": ...}} 或二维数组")
    if arr.shape != (grid.ny, grid.nx):
        raise SiteFormatError(
            f"{name} 形状 {arr.shape} 与网格 ({grid.ny},{grid.nx}) 不符"
        )
    return arr


def load_grid_csv(path: str | Path) -> np.ndarray:
    """读纯数值矩阵（ny 行 × nx 列）；容忍一行表头。"""
    path = Path(path)
    if not path.exists():
        raise SiteFormatError(f"找不到矩阵文件：{path}")
    rows: list[list[float]] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for lineno, raw in enumerate(csv.reader(fh), start=1):
            cells = [c.strip() for c in raw if c.strip() != ""]
            if not cells:
                continue
            try:
                rows.append([float(c) for c in cells])
            except ValueError:
                if lineno == 1:
                    continue          # 表头
                raise SiteFormatError(f"{path}:{lineno} 含非数值：{cells[:3]}") from None
    if not rows:
        raise SiteFormatError(f"{path} 是空文件")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise SiteFormatError(f"{path} 各行长度不一致：{sorted(widths)}")
    return np.asarray(rows, dtype=float)


def load_boreholes_csv(path: str | Path) -> tuple[Borehole, ...]:
    """读钻孔表：x,y,ground,top,bottom,material[,source]，每行一层。"""
    path = Path(path)
    if not path.exists():
        raise SiteFormatError(f"找不到钻孔文件：{path}")
    grouped: dict[tuple[float, float], dict] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"x", "y", "ground", "top", "bottom", "material"}
        if not reader.fieldnames or not required.issubset(
            {f.strip() for f in reader.fieldnames}
        ):
            raise SiteFormatError(
                f"{path} 需要表头 {sorted(required)}，实际 {reader.fieldnames}"
            )
        for lineno, row in enumerate(reader, start=2):
            try:
                x = float(row["x"])
                y = float(row["y"])
                ground = float(row["ground"])
                top = float(row["top"])
                bottom = float(row["bottom"])
                key = row["material"].strip()
            except (TypeError, ValueError):
                raise SiteFormatError(f"{path}:{lineno} 数值列无法解析") from None
            if bottom <= top:
                raise SiteFormatError(f"{path}:{lineno} bottom 必须大于 top")
            if key not in materials.LIBRARY:
                raise SiteFormatError(
                    f"{path}:{lineno} 未知材料 {key!r}；可用："
                    f"{', '.join(sorted(materials.LIBRARY))}"
                )
            slot = grouped.setdefault((x, y), {"ground": ground, "layers": [],
                                               "source": (row.get("source") or "survey").strip()})
            slot["layers"].append((top, bottom, key))

    boreholes = []
    for (x, y), slot in grouped.items():
        layers = sorted(slot["layers"], key=lambda t: t[0])
        seq, cursor = [], 0.0
        for top, bottom, key in layers:
            if top > cursor + 1e-6:
                raise SiteFormatError(
                    f"钻孔 ({x:.1f},{y:.1f}) 在 {cursor:.2f}–{top:.2f} m 处有缺失层"
                )
            seq.append((max(bottom - top, 1e-3), key))
            cursor = bottom
        boreholes.append(Borehole(x, y, slot["ground"], tuple(seq), slot["source"]))
    if not boreholes:
        raise SiteFormatError(f"{path} 没有数据行")
    return tuple(boreholes)


def _borehole_from_dict(d: dict) -> Borehole:
    try:
        return Borehole(float(d["x"]), float(d["y"]), float(d["ground"]),
                        tuple((float(t), str(k)) for t, k in d["layers"]),
                        d.get("source", "survey"))
    except (KeyError, TypeError, ValueError) as exc:
        raise SiteFormatError(f"钻孔定义不合法：{d}") from exc


# ---------------------------------------------------------------------------
# 写
# ---------------------------------------------------------------------------
def save_site(spec: SiteSpec, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = spec.to_dict(inline_arrays=True)
    if doc.get("elevation") is not None:
        doc["elevation"] = np.round(np.asarray(doc["elevation"], dtype=float), 3).tolist()
    doc["temp"] = None if spec.temp is None else np.round(spec.temp, 2).tolist()
    doc["precip"] = None if spec.precip is None else np.round(spec.precip, 1).tolist()
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def write_site_template(directory: str | Path, *, nx: int = 8, ny: int = 8,
                        cell: float = 100.0) -> list[Path]:
    """生成一套最小的可编辑模板（site.json + 三个 CSV），用于接入真实数据。"""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    dem = np.linspace(10.0, 30.0, nx * ny).reshape(ny, nx)
    temp = np.full((ny, nx), 13.0)
    precip = np.full((ny, nx), 800.0)
    np.savetxt(d / "dem.csv", dem, delimiter=",", fmt="%.2f")
    np.savetxt(d / "temp.csv", temp, delimiter=",", fmt="%.2f")
    np.savetxt(d / "precip.csv", precip, delimiter=",", fmt="%.1f")

    with (d / "boreholes.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "ground", "top", "bottom", "material", "source"])
        for x in (cell * 2, cell * (nx - 2)):
            for y in (cell * 2, cell * (ny - 2)):
                w.writerow([x, y, 20.0, 0.0, 1.0, "silt", "survey"])
                w.writerow([x, y, 20.0, 1.0, 4.0, "clay_soft", "survey"])
                w.writerow([x, y, 20.0, 4.0, 9.0, "sand_dense", "survey"])

    doc = {
        "grid": {"nx": nx, "ny": ny, "cell": cell, "x0": 0, "y0": 0},
        "gravity": 9.81,
        "seed": 1,
        "elevation": {"csv": "dem.csv"},
        "temp": {"csv": "temp.csv"},
        "precip": {"csv": "precip.csv"},
        "boreholes": {"csv": "boreholes.csv"},
        "provenance": {"source": "template", "note": "用真实数据替换 CSV 即可"},
    }
    site_path = d / "site.json"
    site_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return [site_path, d / "dem.csv", d / "temp.csv", d / "precip.csv",
            d / "boreholes.csv"]
