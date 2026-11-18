"""一个只回答固定 JSON 的假 Ollama 服务：没有本地模型时也能演示 LLM 路径。

    python tools/fake_ollama.py --port 11500
    # 另一个终端
    blockworld agent --base-url http://127.0.0.1:11500 --model qwen2.5:7b

支持 `GET /api/tags` 与 `POST /api/chat`，行为与 Ollama 一致；只用于验证链路与
解析，不产生任何真实推理。
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

DEFAULT_ADVICE = {
    "summary": "扫描纠正了材料误判，换填后判据通过。",
    "verdict": "certify",
    "risks": ["未测区块仍依赖先验"],
    "recommendations": ["对基础范围补 3 个 CPT 点"],
    "caveats": ["结论依赖 4σ 门控未误杀"],
    "confidence": 0.66,
}


def make_handler(advice: dict, model: str):
    class Handler(BaseHTTPRequestHandler):
        calls: list[dict] = []

        def log_message(self, *a):        # 安静
            return

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/tags":
                self._json({"models": [{"name": model}]})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            Handler.calls.append(payload)
            if self.path == "/api/chat":
                self._json({"message": {"role": "assistant",
                                        "content": json.dumps(advice, ensure_ascii=False)},
                            "done": True})
            else:
                self._json({"error": "not found"}, 404)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser("fake-ollama")
    ap.add_argument("--port", type=int, default=11500)
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--advice", default="", help="自定义 JSON 回答")
    args = ap.parse_args()

    advice = json.loads(args.advice) if args.advice else DEFAULT_ADVICE
    srv = HTTPServer(("127.0.0.1", args.port), make_handler(advice, args.model))
    print(f"fake ollama on http://127.0.0.1:{args.port} (model={args.model})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
