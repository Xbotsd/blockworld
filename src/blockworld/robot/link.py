"""链路：机器人与计算端之间的传输可替换。

"终端替换"有两层含义，这里都覆盖：

* **载荷（sensor payload）替换** —— 见 `devices.py`，换的是扫描头；
* **链路（transport）替换** —— 本模块，换的是数据怎么过来：进程内回环、
  JSONL 录制/回放、TCP 行协议。上层 `RobotSession` 不变。

一台真实机器人通常只需要实现 `Transport`：`send()` 下发指令，`recv()` 返回
一行 JSON（例如 `{"kind": "elevation", "x": 12.0, ...}`）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Protocol


class Transport(Protocol):
    """最小链路契约。"""

    name: str

    def open(self) -> None: ...

    def send(self, event: dict) -> None: ...

    def recv(self) -> dict | None: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# 进程内回环：默认，用于仿真与测试
# ---------------------------------------------------------------------------
class Loopback:
    name = "loopback"

    def __init__(self) -> None:
        self.events: list[dict] = []
        self._inbox: list[dict] = []

    def open(self) -> None:
        return None

    def send(self, event: dict) -> None:
        self.events.append(event)
        self._inbox.append(event)

    def recv(self) -> dict | None:
        return self._inbox.pop(0) if self._inbox else None

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# JSONL 录制 / 回放
# ---------------------------------------------------------------------------
class JsonlWriter:
    """把事件流写到文件；之后可以用回放链路原样重跑。"""

    name = "jsonl"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = None

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def send(self, event: dict) -> None:
        if self._fh is None:
            raise RuntimeError("链路未打开")
        self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def recv(self) -> dict | None:
        return None

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class JsonlReplay:
    """把录制的事件流当作一台"终端"喂回来。"""

    name = "replay"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._events: list[dict] = []
        self._cursor = 0

    def open(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"回放文件不存在：{self.path}")
        with self.path.open(encoding="utf-8-sig") as fh:
            self._events = [json.loads(line) for line in fh if line.strip()]
        self._cursor = 0

    def send(self, event: dict) -> None:
        return None

    def recv(self) -> dict | None:
        if self._cursor >= len(self._events):
            return None
        ev = self._events[self._cursor]
        self._cursor += 1
        return ev

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
TransportFactory = Callable[[str], Transport]

TRANSPORTS: dict[str, TransportFactory] = {
    "loopback": lambda spec: Loopback(),
    "jsonl": lambda spec: JsonlWriter(spec.split(":", 1)[1]),
    "replay": lambda spec: JsonlReplay(spec.split(":", 1)[1]),
}


def register_transport(name: str, factory: TransportFactory) -> None:
    """替换链路。spec 形如 ``"tcp:192.168.1.10:9000"``。"""
    TRANSPORTS[name] = factory


def make_transport(spec: str = "loopback") -> Transport:
    """``spec`` 形如 ``loopback`` / ``jsonl:out/run.jsonl`` / ``replay:out/run.jsonl``。"""
    name = spec.split(":", 1)[0]
    if name not in TRANSPORTS:
        raise KeyError(f"未知链路 {name!r}；可用：{', '.join(sorted(TRANSPORTS))}")
    return TRANSPORTS[name](spec)
