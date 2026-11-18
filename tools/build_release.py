"""打包发布：dist/blockworld-<版本>.zip（源码 + 文档 + 中文手册）。

排除：版本库、构建产物、运行输出、缓存。解压后得到一个顶层目录，直接可跑。
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"

EXCLUDE_DIRS = {
    ".git", ".github_runs", "dist", "__pycache__", ".pytest_cache",
    ".venv", "venv", "build", "out", "out_study", "out_robot", "out_agent",
    "out_smoke", "out_probe", ".idea", ".vscode",
}
EXCLUDE_PREFIX = (".dsh-office-",)
EXCLUDE_SUFFIX = (".pyc", ".pyo", ".egg-info")


def version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    if not m:
        raise SystemExit("pyproject.toml 里没有 version")
    return m.group(1)


def include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if any(p in EXCLUDE_DIRS for p in parts[:-1]):
        return False
    if any(parts[-1].startswith(p) for p in EXCLUDE_PREFIX):
        return False
    if any(parts[-1].endswith(s) for s in EXCLUDE_SUFFIX):
        return False
    if parts[-1] in EXCLUDE_DIRS:
        return False
    return True


def build() -> Path:
    v = version()
    DIST.mkdir(exist_ok=True)
    target = DIST / f"blockworld-v{v}.zip"
    top = f"blockworld-{v}"
    files = [p for p in ROOT.rglob("*") if p.is_file() and include(p)]
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(files):
            arc = f"{top}/{p.relative_to(ROOT).as_posix()}"
            zf.write(p, arc)
    size = target.stat().st_size
    print(f"written: {target}")
    print(f"files:   {len(files)}")
    print(f"size:    {size / 1024:.1f} KiB")
    return target


if __name__ == "__main__":
    build()
