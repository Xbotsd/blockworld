"""一个可复现的闭环场景：先验被误判，机器人扫出来才知道。

场景设置（写死在代码里，便于复现与对照）：

* **真值场地**：合成分层地质，河谷里是软黏土（c≈12 kPa，E≈4 MPa）。
* **先验模型**：同一天地，但早期勘察把软黏土记成了硬黏土（c≈45，E≈25）——
  这是真实工程里最常见的错误之一：把一层土的名字写错。
* **判据**：1.5×1.5 m 基础、400 kN 荷载。先验下通过（裕度 +0.07），
  真值下不通过（裕度 −1.54）。
* **机器人**：LiDAR 测高程、CPT 测强度、探针测水位，只在基础选址区扫描。

于是闭环的看点不是"模型算得准不准"，而是：**扫描之后，模型改判，机器人决定处理。**
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import materials
from ..api import Intervention, choose_foundation_site
from ..verifiers import Foundation
from ..world import Region, WorldConfig
from .devices import make_device
from .session import RobotSession


@dataclass
class Scenario:
    session: RobotSession
    region: Region
    truth: object
    model: object
    narrative: dict


def make_scenario(
    *,
    seed: int = 5,
    nx: int = 24,
    ny: int = 24,
    n_boreholes: int = 10,
    load_kN: float = 400.0,
    misclassify: str = "clay_soft→clay_stiff",
    session_seed: int = 3,
    treats: bool = True,
    act: bool = True,
) -> Scenario:
    from ..site import build_from, synthesize

    site = synthesize(WorldConfig(nx=nx, ny=ny, n_boreholes=n_boreholes, seed=seed))
    truth = build_from(site)
    model = build_from(site)

    src, _, dst = misclassify.partition("→")
    wrong = materials.get(dst)
    for t in model.tiles:
        t.col.layers = [(th, wrong if m.key == src else m) for th, m in t.col.layers]

    from ..api import Criteria

    criteria = Criteria(purposes=("infrastructure",),
                        foundation=Foundation(B=1.5, L=1.5, D=1.5,
                                              load_kN=load_kN),
                        slope_limit_deg=30.0)
    region = choose_foundation_site(truth, max_slope=8.0, max_elev=60.0, radius=200.0)
    criteria.foundation_site = region

    treatments = ()
    if treats:
        treatments = (
            Intervention("换填 6 m 砂砾", [
                {"op": "material", "material": "gravel_dense",
                 "top_depth": 0, "bottom_depth": 6,
                 "region": region.to_dict()}]),
        )

    session = RobotSession(
        model=model,
        truth=truth,
        criteria=criteria,
        region=region,
        devices=[make_device("lidar", per_scan=6),
                 make_device("cpt", per_scan=3),
                 make_device("probe")],
        seed=session_seed,
        treatments=treatments,
        max_ticks=12,
        act=act,
    )
    return Scenario(
        session=session, region=region, truth=truth, model=model,
        narrative={
            "prior": misclassify,
            "design": f"1.5×1.5 m, {load_kN:.0f} kN",
            "scan_tiles": len(model.region_tiles(region)),
        },
    )
