"""离线大模型客户端：只依赖标准库，默认指向本机。

支持三种后端，都可用 `make_client()` 构造：

    ollama    http://127.0.0.1:11434             （Ollama）
    openai    http://127.0.0.1:8080/v1           （llama.cpp / vLLM / LM Studio）
    scripted  无网络，按预设回答（测试与演示）

设计约束：
  * 没有网络也能跑——`available()` 为假时上层直接走确定性回退；
  * 超时可配，任何异常都转成 `None`，绝不让模型拖垮主流程；
  * 不做流式、不做工具调用循环：一次请求一次回答，便于复现与审计。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol

Poster = Callable[[str, dict, dict, float], dict]


class LLMClient(Protocol):
    name: str
    model: str

    def available(self) -> bool: ...

    def chat(self, messages: list[dict], *, json_mode: bool = False) -> str | None: ...


# ---------------------------------------------------------------------------
# 默认的 HTTP 实现（可注入，便于离线测试）
# ---------------------------------------------------------------------------
def _http_post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get(url: str, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
@dataclass
class OllamaClient:
    model: str = "qwen2.5:7b"
    base_url: str = "http://127.0.0.1:11434"
    timeout: float = 90.0
    name: str = "ollama"
    post: Poster = field(default=_http_post, repr=False)
    get: Callable[[str, dict, float], dict] = field(default=_http_get, repr=False)

    def available(self) -> bool:
        try:
            data = self.get(f"{self.base_url}/api/tags", {}, min(self.timeout, 3.0))
            return bool(data.get("models") is not None)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return False

    def models(self) -> list[str]:
        try:
            data = self.get(f"{self.base_url}/api/tags", {}, min(self.timeout, 3.0))
            return [m.get("name", "") for m in data.get("models", [])]
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return []

    def chat(self, messages: list[dict], *, json_mode: bool = False) -> str | None:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 700},
        }
        if json_mode:
            payload["format"] = "json"
        try:
            data = self.post(f"{self.base_url}/api/chat", payload,
                             {"Content-Type": "application/json"}, self.timeout)
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return None
        msg = data.get("message") or {}
        content = msg.get("content")
        return content if isinstance(content, str) else None


# ---------------------------------------------------------------------------
# OpenAI 兼容（llama.cpp server / vLLM / LM Studio）
# ---------------------------------------------------------------------------
@dataclass
class OpenAICompatClient:
    model: str = "local"
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = "offline"
    timeout: float = 90.0
    name: str = "openai"
    post: Poster = field(default=_http_post, repr=False)

    def _headers(self) -> dict:
        return {"Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"}

    def available(self) -> bool:
        try:
            data = self.post(f"{self.base_url}/chat/completions",
                             {"model": self.model,
                              "messages": [{"role": "user", "content": "ping"}],
                              "max_tokens": 1}, self._headers(), min(self.timeout, 5.0))
            return bool(data.get("choices"))
        except (urllib.error.URLError, OSError, ValueError, TimeoutError):
            return False

    def chat(self, messages: list[dict], *, json_mode: bool = False) -> str | None:
        payload = {"model": self.model, "messages": messages, "temperature": 0.2,
                   "max_tokens": 700}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            data = self.post(f"{self.base_url}/chat/completions", payload,
                             self._headers(), self.timeout)
            return data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, OSError, ValueError, KeyError,
                IndexError, TimeoutError):
            return None


# ---------------------------------------------------------------------------
# 无网络：脚本化 / 空
# ---------------------------------------------------------------------------
class ScriptedClient:
    """按顺序返回预设文本；用于测试与"演示模式"。"""

    name = "scripted"

    def __init__(self, replies: list[str], *, model: str = "scripted",
                 up: bool = True) -> None:
        self.replies = list(replies)
        self.model = model
        self._up = up
        self.calls: list[list[dict]] = []

    def available(self) -> bool:
        return self._up

    def chat(self, messages: list[dict], *, json_mode: bool = False) -> str | None:
        self.calls.append(messages)
        if not self.replies:
            return None
        return self.replies.pop(0)


class NullClient:
    """永远不可用：用来验证"没有模型也能跑完"。"""

    name = "none"
    model = "none"

    def available(self) -> bool:
        return False

    def chat(self, messages: list[dict], *, json_mode: bool = False) -> str | None:
        return None


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
CLIENTS = {
    "ollama": OllamaClient,
    "openai": OpenAICompatClient,
    "none": NullClient,
}


def register_client(name: str, factory) -> None:
    CLIENTS[name] = factory


def make_client(kind: str = "ollama", **kwargs) -> LLMClient:
    if kind not in CLIENTS:
        raise KeyError(f"未知后端 {kind!r}；可用：{', '.join(sorted(CLIENTS))}")
    factory = CLIENTS[kind]
    if factory is NullClient:
        return NullClient()
    return factory(**kwargs)


def describe(client: LLMClient) -> dict:
    return {"backend": client.name, "model": client.model,
            "available": client.available()}
