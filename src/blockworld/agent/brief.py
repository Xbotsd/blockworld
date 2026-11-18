"""把事实包与顾问意见渲染成一份可归档的 Markdown 简报。"""
from __future__ import annotations

from datetime import datetime

from .advisor import Advice
from .context import Facts

ACTION_ZH = {
    "certify": "认证",
    "treat": "处理",
    "rescan": "复扫",
    "abort": "中止",
    "escalate": "转人工",
}


def _fmt_margin(v: float) -> str:
    return f"{v:+.2f}"


def render_brief(facts: Facts, advice: Advice, *, title: str = "现场简报") -> str:
    scen = facts.scenario or {}
    final = facts.final.get("action", "—")
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"生成时间：{datetime.now():%Y-%m-%d %H:%M}　"
                 f"顾问来源：`{advice.source}`"
                 + ("　（**仅建议，不参与判定**）" if advice.advisory_only else ""))
    lines.append("")

    lines.append("## 结论")
    lines.append("")
    lines.append(f"- 确定性判定：**{ACTION_ZH.get(final, final)}**（`{final}`）")
    if advice.verdict and advice.verdict != final:
        lines.append(f"- 模型意见：`{advice.verdict}` —— **与策略不一致，以策略为准**")
    lines.append(f"- {advice.summary}")
    lines.append("")

    if scen:
        lines.append("## 场景")
        lines.append("")
        for k, v in scen.items():
            lines.append(f"- {k}：{v}")
        lines.append("")

    lines.append("## 判据")
    lines.append("")
    lines.append("| 判据 | 先验 | 最终 |")
    lines.append("|---|---|---|")
    last = facts.timeline[-1]["margins"] if facts.timeline else {}
    for key, before in (facts.prior or {}).items():
        b = f"{'通过' if before.get('passed') else '不通过'} ({_fmt_margin(before.get('margin', 0))})"
        after = last.get(key)
        a = "—" if after is None else (
            ("通过" if after >= 0 else "不通过") + f" ({_fmt_margin(after)})")
        lines.append(f"| {key} | {b} | {a} |")
    lines.append("")

    if facts.timeline:
        lines.append("## 闭环时间线")
        lines.append("")
        lines.append("| tick | 覆盖 | 采纳/拒绝 | 判据裕度 | 决策 |")
        lines.append("|---|---|---|---|---|")
        for t in facts.timeline:
            margins = " ".join(f"{k}={_fmt_margin(v)}" for k, v in t["margins"].items())
            lines.append(f"| {t['tick']} | {t['coverage']:.0%} | "
                         f"{t['accepted']}/{t['rejected']} | {margins} | "
                         f"`{t['action']}` |")
        lines.append("")

    treatments = [t for t in facts.timeline if t.get("treatment")]
    if treatments:
        lines.append("## 处置")
        lines.append("")
        for t in treatments:
            state = "已执行" if t.get("applied") else "仅建议"
            lines.append(f"- tick {t['tick']}：**{t['treatment']}**（{state}）")
        lines.append("")

    err = facts.model_error
    if err.get("before") and err.get("at_decision"):
        b, a = err["before"], err["at_decision"]
        lines.append("## 模型修正")
        lines.append("")
        lines.append(f"- 黏聚力平均误差：{b.get('cohesion_mae_kpa')} → "
                     f"{a.get('cohesion_mae_kpa')} kPa")
        lines.append(f"- 摩擦角平均误差：{b.get('phi_mae_deg')} → {a.get('phi_mae_deg')}°")
        lines.append(f"- 材料错判率：{b.get('material_mismatch')} → "
                     f"{a.get('material_mismatch')}")
        lines.append("")

    for section, items in (("风险", advice.risks),
                           ("建议", advice.recommendations),
                           ("前提与数据缺口", advice.caveats)):
        if not items:
            continue
        lines.append(f"## {section}")
        lines.append("")
        for it in items:
            lines.append(f"- {it}")
        lines.append("")

    lines.append("## 声明")
    lines.append("")
    lines.append("本简报由离线模型生成，仅用于解释与建议。判据、裕度与最终决策来自"
                 "确定性规则与实测同化，模型不具备否决权；模型不可用时本节内容由"
                 "确定性摘要替代。")
    lines.append("")
    return "\n".join(lines)
