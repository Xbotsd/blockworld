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
