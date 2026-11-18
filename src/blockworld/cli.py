"""命令行入口。

    python -m blockworld demo    --out out --years 55        # 端到端演示 + 报告
    python -m blockworld study   study.json --out out        # 跑一个研究（可用 --mc / --morris）
    python -m blockworld template  data/site                 # 生成可编辑的数据模板
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import demo, io, uncertainty
from .api import Criteria, Intervention, Study
from .site import SiteSpec, build_from


def _criteria_from(doc: dict | None) -> Criteria:
    from . import verifiers as V

    doc = doc or {}
    crit = Criteria()
    if "purposes" in doc:
        crit.purposes = tuple(doc["purposes"])
    if "foundation" in doc:
        crit.foundation = V.Foundation(**doc["foundation"])
    if "lander" in doc:
        crit.lander = V.Lander(**doc["lander"])
    for key in ("slope_fs_required", "slope_limit_deg", "eco_min_vegetated",
                "eco_min_biomes", "eco_animal_years"):
        if key in doc:
            setattr(crit, key, doc[key])
    return crit


def cmd_demo(args) -> int:
    summary = demo.run(args.out, spin_years=args.years,
                       animal_years=args.animals, seed=args.seed)
    print(json.dumps({
        "verdicts": {v["title"]: v["passed"] for v in summary["verdicts"]},
        "top_factors": summary["sensitivity"][:3],
        "outputs": [str(Path(args.out) / n)
                    for n in ("report.html", "sandbox.png", "summary.json")],
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_study(args) -> int:
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    base = cfg_path.parent

    site_path = Path(cfg["site"])
    if not site_path.is_absolute():
        site_path = base / site_path
    site = io.load_site(site_path)

    study = Study(
        site=site,
        criteria=_criteria_from(cfg.get("criteria")),
        interventions=tuple(Intervention.from_dict(d)
                            for d in cfg.get("interventions", [])),
        spin_years=int(cfg.get("spin_years", 0)),
        seed=int(cfg.get("seed", 0)),
    )
    result = study.run()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "study.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    payload = {"passed": result.passed(), "flips": result.flips()}

    params = cfg.get("parameters") or []
    if params:
        parsed = [
            uncertainty.Parameter(
                name=p["name"],
                target=p["target"],
                dist=uncertainty.Distribution(**p["dist"]),
                label=p.get("label", ""),
            )
            for p in params
        ]
        iv = None
        if cfg.get("interventions"):
            iv = Intervention.from_dict(cfg["interventions"][0])

        if args.mc:
            mc = uncertainty.monte_carlo(study, parsed, n=args.mc,
                                         seed=int(cfg.get("seed", 0)), intervention=iv)
            (out / "montecarlo.json").write_text(
                json.dumps(mc.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            payload["monte_carlo"] = {"baseline": mc.baseline,
                                      "intervention": mc.intervention}

        if args.morris:
            mr = uncertainty.morris(study, parsed, r=args.morris,
                                    seed=int(cfg.get("seed", 0)), intervention=iv)
            (out / "morris.json").write_text(
                json.dumps(mr.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            top = mr.ranking(args.criterion, use="intervention" if iv else "baseline")
            payload["morris_top"] = top[:5]

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_robot(args) -> int:
    from .robot import make_transport
    from .robot.demo import make_scenario

    sc = make_scenario(seed=args.seed, session_seed=args.session_seed,
                       act=not args.no_act)
    if args.transport:
        sc.session.transport = make_transport(args.transport)
        sc.session.transport.open()

    report = sc.session.run(ticks=args.ticks)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["narrative"] = sc.narrative
    (out / "robot_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for rec in report.ticks:
        vs = " ".join(f"{v['key']}={v['margin']:+.2f}" for v in rec["verdicts"])
        rows.append(f"  t{rec['tick']:<2} 测量 {rec['measurements']:>2} "
                    f"采纳 {rec['accepted']:>2} 覆盖率 {rec['coverage']:.0%}  "
                    f"{vs}  → {rec['decision']['action']}")
    print("\n".join(rows))
    print(f"\n先验：{sc.narrative['prior']}　设计：{sc.narrative['design']}")
    print(f"最终判断：{report.final.action} — {report.final.reason}")
    before = report.meta["model_error_before"]
    at_dec = report.meta.get("model_error_at_decision")
    print(f"模型误差（扫描范围内）先验：{before}")
    if at_dec:
        print(f"                　　　 扫描后（处理前）：{at_dec}")
    if report.meta.get("acted"):
        print("　处理过的区块已排除在误差统计外（模型此时本应偏离自然地面）")
    print(f"\n输出：{out / 'robot_report.json'}")
    return 0


def cmd_agent(args) -> int:
    from .agent import Advisor, Facts, describe, make_client, render_brief
    from .robot.demo import make_scenario

    client = make_client(args.backend, model=args.model, base_url=args.base_url,
                         timeout=args.timeout)
    info = describe(client)

    if args.list_models:
        models = client.models() if hasattr(client, "models") else []
        print(json.dumps({"backend": info["backend"], "available": info["available"],
                          "models": models}, ensure_ascii=False, indent=2))
        return 0

    sc = make_scenario(seed=args.seed, session_seed=args.session_seed,
                       act=not args.no_act)
    report = sc.session.run(ticks=args.ticks)

    facts = Facts.from_report(report, scenario=sc.narrative)
    advice = Advisor(client).advise(facts)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "robot_report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    payload = advice.to_dict()
    payload["policy_action"] = facts.final.get("action")
    (out / "agent_advice.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "agent_brief.md").write_text(
        render_brief(facts, advice), encoding="utf-8")

    print(f"后端：{info['backend']}/{info['model']}　可用：{info['available']}")
    print(f"确定性判定：{facts.final.get('action')}　顾问来源：{advice.source}")
    if advice.verdict and advice.verdict != facts.final.get("action"):
        print(f"注意：模型意见 {advice.verdict} 与策略不一致，以策略为准")
    print(f"\n{advice.summary}\n")
    for title, items in (("风险", advice.risks), ("建议", advice.recommendations),
                         ("前提", advice.caveats)):
        if items:
            print(f"{title}：")
            for it in items:
                print(f"  - {it}")
    print(f"\n输出：{out / 'agent_brief.md'}")
    return 0


def cmd_template(args) -> int:
    files = io.write_site_template(args.dir, nx=args.nx, ny=args.ny, cell=args.cell)
    print(json.dumps({"written": [str(f) for f in files]}, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认不是 UTF-8，帮助文本里的中文会抛 UnicodeEncodeError
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser("blockworld", description="区块化地质—生态沙盘")
    sub = ap.add_subparsers(dest="cmd")

    d = sub.add_parser("demo", help="端到端演示：建世界 → 推演 → 验证 → 替换 → 报告")
    d.add_argument("--out", default="out")
    d.add_argument("--years", type=int, default=55)
    d.add_argument("--animals", type=int, default=6)
    d.add_argument("--seed", type=int, default=None)
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("study", help="按配置跑一个研究（基线 + 干预 + 可选不确定性）")
    s.add_argument("config")
    s.add_argument("--out", default="out")
    s.add_argument("--mc", type=int, default=0, help="Monte Carlo 抽样数")
    s.add_argument("--morris", type=int, default=0, help="Morris 轨迹数 r")
    s.add_argument("--criterion", default="landing", help="Morris 排序依据的判据")
    s.set_defaults(func=cmd_study)

    r = sub.add_parser("robot", help="机器人在环：扫描 → 同化 → 自主判断")
    r.add_argument("--out", default="out_robot")
    r.add_argument("--ticks", type=int, default=8)
    r.add_argument("--seed", type=int, default=5, help="场地种子")
    r.add_argument("--session-seed", type=int, default=3, help="测量噪声种子")
    r.add_argument("--transport", default="",
                   help="loopback / jsonl:路径 / replay:路径")
    r.add_argument("--no-act", action="store_true",
                   help="只给建议，不在模型上执行处理")
    r.set_defaults(func=cmd_robot)

    a = sub.add_parser("agent", help="离线顾问：本地模型解释闭环结果（不参与判定）")
    a.add_argument("--out", default="out_agent")
    a.add_argument("--backend", default="ollama", choices=["ollama", "openai", "none"])
    a.add_argument("--model", default="qwen2.5:7b")
    a.add_argument("--base-url", default="http://127.0.0.1:11434")
    a.add_argument("--timeout", type=float, default=90.0)
    a.add_argument("--ticks", type=int, default=12)
    a.add_argument("--seed", type=int, default=5)
    a.add_argument("--session-seed", type=int, default=3)
    a.add_argument("--no-act", action="store_true")
    a.add_argument("--list-models", action="store_true")
    a.set_defaults(func=cmd_agent)

    t = sub.add_parser("template", help="生成可编辑的场地数据模板")
    t.add_argument("dir")
    t.add_argument("--nx", type=int, default=8)
    t.add_argument("--ny", type=int, default=8)
    t.add_argument("--cell", type=float, default=100.0)
    t.set_defaults(func=cmd_template)

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 1
    return args.func(args)
