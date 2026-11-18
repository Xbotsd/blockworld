"""自主判断：把判据、证据量和覆盖度变成一条可审计的决策。

这里没有黑箱。规则是显式的、可读的、可改阈值的：

    1. 证据不足（测量数 < min_evidence 或覆盖度 < min_coverage） → 复扫
    2. 有判据不通过，且存在能改善它的处理方案              → 处理
    3. 有判据不通过，且没有可用方案                        → 中止
    4. 全部通过但裕度贴近阈值（|margin| < near_threshold） → 复扫
    5. 全部通过且裕度充足                                  → 认证

这样回答的是"机器人能不能自己下结论"，而不是"模型输出一个分数"：结论必须
同时满足**判据**、**证据量**和**空间覆盖**三个条件，缺一个就退回复扫。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..verifiers import Verdict
from .types import Decision


@dataclass
class PolicyConfig:
    certify_margin: float = 0.30    # 认证所需的最小裕度（0 ≤ margin < 该值视为"不足以认证"）
    min_evidence: int = 12          # 至少多少条有效测量
    min_coverage: float = 0.90      # 至少覆盖多少比例的扫描区（按地下证据计）
    p_fail_limit: float = 0.50      # 失效概率超过则不允许认证

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class Supervisor:
    """规则型决策器：输入判据与证据，输出行动。"""

    cfg: PolicyConfig = field(default_factory=PolicyConfig)

    def decide(
        self,
        verdicts: list[Verdict],
        *,
        coverage: float,
        evidence: int,
        has_treatments: bool = False,
        p_fail: dict[str, float] | None = None,
        stagnant: bool = False,
        tick: int = 0,
    ) -> Decision:
        c = self.cfg
        if not verdicts:
            return Decision("abort", "没有判据可评估", 0.0, tick=tick)

        weakest = min(verdicts, key=lambda v: v.margin)
        failing = [v for v in verdicts if not v.passed]

        # 1) 证据不足 → 复扫（明确说缺什么）
        if evidence < c.min_evidence or coverage < c.min_coverage:
            need_ev = max(0, c.min_evidence - evidence)
            need_cov = max(0.0, c.min_coverage - coverage)
            conf = min(1.0, coverage / max(c.min_coverage, 1e-6)) * 0.5
            return Decision(
                "rescan",
                f"证据不足：有效测量 {evidence}/{c.min_evidence}，"
                f"覆盖 {coverage * 100:.0f}%/{c.min_coverage * 100:.0f}%；"
                f"还需约 {need_ev} 条或 {need_cov * 100:.0f}% 覆盖",
                conf,
                {"need_evidence": need_ev, "need_coverage": round(need_cov, 3),
                 "weakest": weakest.key, "weakest_margin": round(weakest.margin, 3)},
                tick,
            )

        # 2) 失效概率过高 → 不允许认证
        if p_fail:
            worst_p = max(p_fail.items(), key=lambda kv: kv[1])
            if worst_p[1] > c.p_fail_limit:
                action = "treat" if has_treatments else "abort"
                return Decision(
                    action,
                    f"{worst_p[0]} 的失效概率 {worst_p[1]:.0%} 超过上限 "
                    f"{c.p_fail_limit:.0%}，结论不可认证",
                    1.0 - worst_p[1],
                    {"p_fail": dict(p_fail)},
                    tick,
                )

        # 3) 有判据不通过
        if failing:
            worst = min(failing, key=lambda v: v.margin)
            if has_treatments:
                return Decision(
                    "treat",
                    f"{worst.title} 未通过（裕度 {worst.margin:.2f}），"
                    f"尝试可用处理方案",
                    min(1.0, c.min_coverage + (coverage - c.min_coverage)),
                    {"failed": [v.key for v in failing],
                     "worst": worst.key, "worst_margin": round(worst.margin, 3)},
                    tick,
                )
            return Decision(
                "abort",
                f"{worst.title} 未通过（裕度 {worst.margin:.2f}），"
                f"且没有可改善判据的处理方案",
                1.0,
                {"failed": [v.key for v in failing], "worst": worst.key},
                tick,
            )

        # 4) 通过但裕度不足以认证
        weak = [v for v in verdicts if v.margin < c.certify_margin]
        if weak:
            w = min(weak, key=lambda v: v.margin)
            if stagnant:
                return Decision(
                    "escalate",
                    f"{w.title} 裕度 {w.margin:.2f} 连续复扫无变化，"
                    f"低于认证线 {c.certify_margin:.2f}；转人工复核",
                    0.3,
                    {"weakest": w.key, "weakest_margin": round(w.margin, 3)},
                    tick,
                )
            return Decision(
                "rescan",
                f"{w.title} 裕度 {w.margin:.2f} 低于认证线 {c.certify_margin:.2f}，"
                f"需要更密的采样才能下结论",
                0.5 + 0.5 * min(1.0, w.margin / max(c.certify_margin, 1e-6)),
                {"weakest": w.key, "weakest_margin": round(w.margin, 3)},
                tick,
            )

        # 5) 通过且裕度充足 → 认证
        conf = min(
            1.0,
            min(1.0, (weakest.margin - c.certify_margin) / 0.5) *
            min(1.0, coverage / max(c.min_coverage, 1e-6)) *
            min(1.0, evidence / max(c.min_evidence * 2, 1)),
        )
        return Decision(
            "certify",
            f"全部 {len(verdicts)} 项判据通过，最弱裕度 {weakest.margin:.2f}，"
            f"覆盖 {coverage * 100:.0f}%，有效测量 {evidence} 条",
            max(0.05, conf),
            {"weakest": weakest.key, "weakest_margin": round(weakest.margin, 3)},
            tick,
        )


def rules() -> list[dict]:
    """把规则表暴露给文档与 CLI，避免文档与实现漂移。"""
    return [
        {"order": 1, "condition": "证据不足", "action": "rescan"},
        {"order": 2, "condition": "p_fail > 上限", "action": "treat / abort"},
        {"order": 3, "condition": "判据不通过且有方案", "action": "treat"},
        {"order": 4, "condition": "判据不通过且无方案", "action": "abort"},
        {"order": 5, "condition": "裕度低于认证线", "action": "rescan / escalate"},
        {"order": 6, "condition": "通过且裕度充足", "action": "certify"},
    ]
