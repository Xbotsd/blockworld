"""离线顾问：把事实包交给本地模型，拿回一份"只能建议、不能否决"的意见。

三条硬约束写进代码而不是文档里：

1. `Advice.advisory_only` 恒为 True——它没有能力修改判据、裕度或决策；
2. 模型给出的 verdict 与确定性策略不一致时，只记录 `agrees_with_policy=False`
   并转成一条 caveat，不改变任何结果；
3. 模型不可用 / 超时 / 输出不合法时，回退到**确定性摘要**（同一份事实，
   不含任何生成内容），保证流水线永远有产出。

这样"接入离线 AI"不会把可审计的工程判断换成一段无法复现的文本。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .context import Facts
from .llm import LLMClient, NullClient

ACTIONS = ("certify", "treat", "rescan", "abort", "escalate")

SYSTEM_PROMPT = """你是一台离线部署的岩土/地质工程审查助手，运行在工地内网，没有任何联网能力。

你会收到一个 JSON 事实包，描述一次"机器人扫描 → 模型同化 → 判据评估 → 决策"的闭环运行。
你只能依据事实包里的数字说话，不要引入外部知识去假设场地条件。

硬性规则：
1. 判据与决策由确定性规则产生，已经是结论。你可以质疑，但只能写进 caveats，
   并说明"缺少哪一项数据才能改变结论"。
2. verdict 字段必须与事实包中的 final.action 一致；如果你认为不一致，仍填写
   两者中你认为更合理的那个，但必须在 caveats 里写明理由。
3. margin 为负表示不通过；margin 是归一化裕度，不能当作安全系数换算。
4. 不要编造未给出的测量值、材料参数或规范条文编号。
5. 输出必须是单个 JSON 对象，不要代码块，不要多余文字。

JSON 结构：
{"summary": "两句话以内的结论",
 "verdict": "certify|treat|rescan|abort|escalate",
 "risks": ["风险，最多 4 条"],
 "recommendations": ["可执行建议，最多 4 条"],
 "caveats": ["结论的前提或数据缺口，最多 4 条"],
 "confidence": 0.0}
"""


@dataclass
class Advice:
    summary: str
    risks: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    confidence: float = 0.0
    verdict: str | None = None
    agrees_with_policy: bool = True
    source: str = "fallback"
    raw: str = ""
    advisory_only: bool = True            # 不可关闭

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "verdict": self.verdict,
            "policy_action": None,          # 由 brief 填充
            "agrees_with_policy": self.agrees_with_policy,
            "risks": self.risks,
            "recommendations": self.recommendations,
            "caveats": self.caveats,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "advisory_only": self.advisory_only,
        }


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_advice(text: str) -> dict | None:
    """容忍代码块、前后废话、尾随逗号；失败返回 None。"""
    if not text:
        return None
    candidates = []
    m = _FENCE.search(text)
    if m:
        candidates.append(m.group(1))
    candidates.append(text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


def _clip(items, limit: int = 4, width: int = 220) -> list[str]:
    out = []
    if not isinstance(items, list):
        return out
    for it in items:
        if isinstance(it, dict):
            it = "；".join(f"{k}: {v}" for k, v in it.items())
        if not isinstance(it, str):
            continue
        it = it.strip()
        if it:
            out.append(it[:width])
        if len(out) >= limit:
            break
    return out


def _validate(data: dict, facts: Facts, *, source: str, raw: str) -> Advice:
    verdict = data.get("verdict")
    if verdict not in ACTIONS:
        verdict = None
    policy_action = facts.final.get("action")
    agree = True if verdict is None else (verdict == policy_action)
    caveats = _clip(data.get("caveats"))
    if not agree:
        caveats.insert(0, f"模型认为应为 {verdict}，确定性策略判定为 {policy_action}；"
                          f"以策略为准，差异已记录")
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return Advice(
        summary=str(data.get("summary", ""))[:600] or "（模型未给出摘要）",
        risks=_clip(data.get("risks")),
        recommendations=_clip(data.get("recommendations")),
        caveats=caveats[:5],
        confidence=max(0.0, min(1.0, conf)),
        verdict=verdict,
        agrees_with_policy=agree,
        source=source,
        raw=raw[:4000],
    )


# ---------------------------------------------------------------------------
# 确定性回退
# ---------------------------------------------------------------------------
def fallback_advice(facts: Facts, *, reason: str) -> Advice:
    """不依赖任何模型的摘要：同一份事实，直接拼出来。"""
    action = facts.final.get("action") or "unknown"
    cov = facts.evidence.get("coverage") or 0.0
    margins = {k: v for k, v in (facts.prior or {}).items()}
    last = facts.timeline[-1]["margins"] if facts.timeline else {}
    weak = min(last.items(), key=lambda kv: kv[1]) if last else (None, 0.0)
    err_b = (facts.model_error.get("before") or {}).get("cohesion_mae_kpa")
    err_a = (facts.model_error.get("at_decision") or {}).get("cohesion_mae_kpa")

    risks = []
    failed = [k for k, v in (facts.prior or {}).items() if not v.get("passed", True)]
    if failed:
        risks.append(f"先验阶段即未通过的判据：{', '.join(failed)}")
    if weak[0] is not None and weak[1] < 0.3:
        risks.append(f"{weak[0]} 的最终裕度仅 {weak[1]:+.2f}，接近认证线")
    if cov and cov < 0.9:
        risks.append(f"扫描覆盖 {cov:.0%}，未覆盖区块仍依赖先验")

    recs = []
    applied = [t for t in facts.timeline if t.get("treatment")]
    if applied:
        key = applied[-1]["treatment"]
        state = "已执行" if applied[-1].get("applied") else "建议执行"
        recs.append(f"{state}「{key}」，并按同强度复扫复核")
    if weak[0] is not None:
        recs.append(f"对 {weak[0]} 增加采样密度，尤其是地下证据（强度/水位）")
    recs.append("把本次事实包与 JSONL 事件流归档，供后续复核")

    caveats = [reason]
    if err_b is not None and err_a is not None:
        caveats.append(f"模型黏聚力误差由 {err_b} kPa 降到 {err_a} kPa，"
                       f"仍非零，说明未测区块保留了先验")
    return Advice(
        summary=(f"确定性策略判定为 {action}；扫描覆盖 {cov:.0%}，"
                 f"最弱判据 {weak[0] or '—'} 裕度 {weak[1]:+.2f}。"),
        risks=risks, recommendations=recs, caveats=caveats,
        confidence=0.0, verdict=action, agrees_with_policy=True,
        source=f"fallback({reason})",
    )


# ---------------------------------------------------------------------------
# 顾问
# ---------------------------------------------------------------------------
@dataclass
class Advisor:
    client: LLMClient = field(default_factory=NullClient)
    system_prompt: str = SYSTEM_PROMPT
    max_facts_chars: int = 8000

    def advise(self, facts: Facts) -> Advice:
        if not self.client.available():
            return fallback_advice(facts, reason="模型不可用，未联网")

        payload = facts.to_text()
        if len(payload) > self.max_facts_chars:
            payload = payload[: self.max_facts_chars] + "…(截断)"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"事实包：\n{payload}\n\n请按要求输出单个 JSON。"},
        ]
        text = self.client.chat(messages, json_mode=True)
        if text is None:
            return fallback_advice(facts, reason="模型无响应或超时")

        data = parse_advice(text)
        if data is None:
            adv = fallback_advice(facts, reason="模型输出不是合法 JSON")
            adv.raw = text[:4000]
            return adv
        return _validate(data, facts, source=f"llm:{self.client.name}/{self.client.model}",
                         raw=text)
