"""离线顾问 agent：本地模型只解释、只建议，不参与判定。

    from blockworld.agent import Advisor, make_client, Facts

    advisor = Advisor(make_client("ollama", model="qwen2.5:7b"))
    advice = advisor.advise(Facts.from_report(session_report))
    advice.summary, advice.risks, advice.source

没有本地模型时 `advice.source` 会是 `fallback(...)`，内容由确定性规则拼出，
主流程不受影响。
"""

from .advisor import (Advice, Advisor, SYSTEM_PROMPT, fallback_advice,
                      parse_advice)
from .brief import render_brief
from .context import Facts
from .llm import (CLIENTS, LLMClient, NullClient, OllamaClient,
                  OpenAICompatClient, ScriptedClient, describe, make_client,
                  register_client)

__all__ = [
    "Advisor", "Advice", "Facts", "render_brief",
    "parse_advice", "fallback_advice", "SYSTEM_PROMPT",
    "make_client", "register_client", "describe",
    "LLMClient", "OllamaClient", "OpenAICompatClient",
    "ScriptedClient", "NullClient", "CLIENTS",
]
