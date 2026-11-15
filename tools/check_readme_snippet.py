"""README 里的示例，逐字跑一遍（防止文档漂移）。"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from blockworld import Criteria, Intervention, Region, Study
from blockworld.io import load_site

study = Study(
    site=load_site("data/site/site.json"),
    criteria=Criteria(
        purposes=("infrastructure", "landing"),
        landing_site=Region.circle(600, 300, 150),
    ),
    interventions=(
        Intervention("换填 8 m 砂砾", [
            {"op": "material", "material": "gravel_dense",
             "top_depth": 0, "bottom_depth": 8, "region": {"kind": "all"}}]),
        Intervention("着陆场胶结 2.5 m", [
            {"op": "material", "material": "cemented_regolith",
             "top_depth": 0, "bottom_depth": 2.5,
             "region": {"kind": "circle", "cx": 600, "cy": 300, "r": 150}}]),
    ),
)
result = study.run()
print(result.passed())
print([(f["intervention"], f["criterion"], f["passed_before"], f["passed_after"])
       for f in result.flips()])
