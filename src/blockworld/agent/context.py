"""把一次闭环运行压成"事实包"，供离线模型阅读。

只给判断需要的东西：判据与裕度、扫描覆盖、决策序列、模型误差、策略阈值。
不塞原始测量（几百条点会淹没上下文），也不塞任何模型不该看到的东西。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..robot import rules as policy_rules


@dataclass
class Facts:
    scenario: dict = field(default_factory=dict)
    prior: dict = field(default_factory=dict)         # criterion -> {passed, margin}
    timeline: list[dict] = field(default_factory=list)
    final: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    model_error: dict = field(default_factory=dict)
    policy: dict = field(default_factory=dict)
    rules: list[dict] = field(default_factory=list)

    # -- 构造 -----------------------------------------------------------
    @staticmethod
    def from_report(report, *, scenario: dict | None = None) -> "Facts":
        d = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        meta = d.get("meta", {})
        timeline = []
        for t in d.get("ticks", []):
            payload = t["decision"].get("payload", {}) or {}
            timeline.append({
                "tick": t["tick"],
                "coverage": t["coverage"],
                "evidence": t["evidence"],
                "accepted": t["accepted"],
                "rejected": t["rejected"],
                "margins": {v["key"]: v["margin"] for v in t["verdicts"]},
                "action": t["decision"]["action"],
                "reason": t["decision"]["reason"],
                "treatment": payload.get("treatment"),
                "applied": bool(payload.get("applied")),
            })
        final = d.get("final") or {}
        return Facts(
            scenario=scenario or meta.get("narrative", {}),
            prior=d.get("before", {}),
            timeline=timeline,
            final={"action": final.get("action"), "reason": final.get("reason"),
                   "confidence": final.get("confidence"),
                   "payload": final.get("payload", {})},
            evidence={
                "measurements": d.get("measurements"),
                "assimilated": d.get("assimilated"),
                "rejected": d.get("rejected"),
                "coverage": d.get("coverage"),
                "scope_tiles": meta.get("scope_tiles"),
                "measured_tiles": meta.get("measured_tiles"),
            },
            model_error={
                "before": meta.get("model_error_before"),
                "after": meta.get("model_error_after"),
                "at_decision": meta.get("model_error_at_decision"),
            },
            policy=meta.get("policy", {}),
            rules=policy_rules(),
        )

    # -- 输出 -----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "prior": self.prior,
            "timeline": self.timeline,
            "final": self.final,
            "evidence": self.evidence,
            "model_error": self.model_error,
            "policy": self.policy,
            "rules": self.rules,
        }

    def to_text(self) -> str:
        """紧凑 JSON：比自然语言更省 token，也更容易被小模型正确解析。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))
