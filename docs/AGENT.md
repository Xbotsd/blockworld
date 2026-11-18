# Offline agent

A decision loop that runs on a construction site has to work with no network.
This module puts a locally deployed model (**Ollama**, `llama.cpp` server,
vLLM, LM Studio — anything OpenAI-compatible) next to the deterministic rules,
with one rule that does not bend:

> **The model explains and advises. It never decides.**

```
RobotSession ──► SessionReport ──► Facts (JSON) ──► local model ──► Advice
                                                     │                │
                                            unavailable/timeout ──────┤
                                                     ▼                ▼
                                              deterministic      brief.md
                                                fallback        (advisory only)
```

## Why it is built this way

An LLM can invent a soil parameter, a code clause, or a factor of safety. In a
geotechnical verdict that is unacceptable, and it is also unnecessary: the
numbers are already computed and auditable. So the model is used for the parts
where language models actually help —

* turning a tick-by-tick trace into a readable site brief,
* naming risks the criteria did not encode (unmeasured zones, a treatment that
  was assumed rather than verified),
* proposing the next action in words an operator can act on.

Everything else stays deterministic.

## Usage

```python
from blockworld.agent import Advisor, Facts, make_client, render_brief

client = make_client("ollama", model="qwen2.5:7b")      # http://127.0.0.1:11434
facts  = Facts.from_report(session_report)
advice = Advisor(client).advise(facts)

advice.summary            # 两句话结论
advice.risks              # 风险
advice.recommendations    # 建议
advice.caveats            # 前提 / 数据缺口
advice.source             # "llm:ollama/qwen2.5:7b" 或 "fallback(模型不可用，未联网)"
advice.agrees_with_policy # 模型意见是否与确定性判定一致
advice.advisory_only      # 恒为 True，不可关闭

open("brief.md", "w", encoding="utf-8").write(render_brief(facts, advice))
```

Command line:

```bash
blockworld agent --out out_agent --model qwen2.5:7b            # Ollama
blockworld agent --backend openai --base-url http://127.0.0.1:8080/v1 \
                 --model local                                  # llama.cpp / LM Studio
blockworld agent --backend none                                # 强制走确定性回退
blockworld agent --list-models                                 # 看看本机装了什么
```

## No platform dependency

There is no vendor SDK, no account, no cloud endpoint and no API-key flow in the
code path. A client is an HTTP POST to a URL you choose:

* speaks the Ollama protocol → `--backend ollama`;
* speaks the OpenAI chat-completions shape (llama.cpp, vLLM, LM Studio, SGLang,
  text-generation-webui, a 30-line Flask shim) → `--backend openai`;
* speaks something else → implement `LLMClient` (`available()` + `chat()`) and
  `register_client("yours", Factory)`. That is the whole contract.

Configuration can come from the environment, so a deployment needs no code change:

```bash
export BLOCKWORLD_LLM_BACKEND=openai
export BLOCKWORLD_LLM_BASE_URL=http://10.0.0.5:8080/v1
export BLOCKWORLD_LLM_MODEL=local-7b
export BLOCKWORLD_LLM_API_KEY=offline          # most local servers ignore it
export BLOCKWORLD_LLM_TIMEOUT=120
blockworld agent --out out_agent                # flags still override env
```

Precedence: command-line flag → environment variable → built-in default. Nothing
is contacted unless `available()` succeeds, and every failure path is a local
fallback.

## Backends

| backend | endpoint | notes |
|---|---|---|
| `ollama` | `POST /api/chat` | `format: "json"` 强制 JSON 输出；`/api/tags` 做健康检查 |
| `openai` | `POST /v1/chat/completions` | llama.cpp server、vLLM、LM Studio；`response_format` 请求 JSON |
| `none` | — | 永不联网；用于测试与"无模型"部署 |

注册新的后端：

```python
from blockworld.agent import register_client
register_client("my-rig", lambda **kw: MyClient(**kw))
```

## Guarantees

1. **No veto.** `Advice.advisory_only` is `True` by construction. `StudyResult`
   and `SessionReport` are never mutated by the advisor.
2. **Disagreement is recorded, not applied.** If the model returns a `verdict`
   different from `final.action`, `agrees_with_policy` becomes `False`, a caveat
   is prepended, and the brief prints both — with the policy marked as
   authoritative.
3. **Fallback always produces a brief.** No model, timeout, or malformed JSON
   yields a deterministic summary built from the same facts, with `source`
   starting `fallback(`. The JSONL/audit trail is unchanged.
4. **Bounded input.** The facts pack is compact JSON (no raw measurement
   streams) and is truncated at `max_facts_chars` so a small local model is not
   drowned.

## Prompt contract

System prompt (abridged, see `agent/advisor.py`):

* you are an offline geotechnical review assistant with no internet;
* speak only from the facts pack; do not import outside assumptions;
* the verdict and margins are already decided — you may question them only in
  `caveats`, and you must say which missing measurement would change them;
* output a single JSON object: `summary`, `verdict`, `risks`,
  `recommendations`, `caveats`, `confidence`.

Parsing tolerates fenced blocks, leading prose and trailing text; anything else
falls back.

## What this does **not** do

* It does not call tools or browse; there is no agentic loop over the repository.
* It does not estimate numbers. Any figure in the brief comes from `Facts`.
* It does not replace a reviewer. `escalate` is still the honest answer when the
  data cannot separate pass from fail.

## No model at hand?

`tools/fake_ollama.py` is a two-endpoint stub that returns a fixed advice JSON. It
exercises the whole path — health check, request shape, JSON parsing, brief — with
no GPU and no download:

```bash
python tools/fake_ollama.py --port 11500
blockworld agent --base-url http://127.0.0.1:11500 --model qwen2.5:7b
```

It deliberately does not reason; use it to verify wiring, not to review a site.

## Model choice

Any instruction-following model with ~7B parameters handles this prompt; the
JSON schema is small and the facts pack is a few kilobytes. Tested shapes:

| model | note |
|---|---|
| `qwen2.5:7b` | good Chinese, follows the JSON schema |
| `llama3.1:8b` | fine, occasionally wraps JSON in prose (parser handles it) |
| `qwen2.5:1.5b` | works, but caveats get generic — fine for a fallback-quality brief |

`blockworld agent --list-models` prints what your local runtime actually has.
