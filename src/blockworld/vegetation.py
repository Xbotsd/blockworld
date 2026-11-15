"""植被推演：队列（cohort）竞争 → 生物群区涌现。

不是"给区块贴一个植被标签"，而是让每个队列按自己的性状（Vcmax、SLA、耐阴、
耐旱、最大高度）去争光争水，长不起来的队列会被死亡率与招募失败淘汰。替换植被
类型 = 改变初始队列与种子库，然后让过程裁决——这就是"该成分在此处是否可持续"
的推演结论。
"""
from __future__ import annotations

import math

from . import pft as pft_mod
from .world import Cohort, Tile, World

GPP_REF = 1.65          # kg C/m²/yr，Vcmax=60 且冠层近饱和时的潜势
RESP_FRAC = 0.48        # 自养呼吸占比
BIOMASS_CAP = 14.0      # kg C/m²


def leaf_fraction(p: pft_mod.PFT) -> float:
    return 0.30 if not p.is_woody else 0.04


def tile_lai(t: Tile) -> float:
    lai = 0.0
    for c in t.cohort:
        p = pft_mod.get(c.pft)
        lai += c.biomass * leaf_fraction(p) * p.sla
    return lai


def _temp_response(p: pft_mod.PFT, temp: float) -> float:
    z = (temp - p.t_opt) / 9.0
    return math.exp(-0.5 * z * z)


def _water_response(p: pft_mod.PFT, t: Tile, pet_veg: float) -> float:
    supply = t.et / max(pet_veg, 1.0)          # 0–1，Budyko 实际/Potential
    if t.wt < 1.5:
        supply = min(1.0, supply + 0.12)       # 浅埋深地下水的补给
    need = 0.62 - 0.30 * p.drought_tol
    return max(0.0, min(1.0, supply / max(need, 0.15)))


def light_profile(t: Tile) -> list[float]:
    """按高度降序返回每个队列截获的光占比。"""
    cohorts = sorted(t.cohort, key=lambda c: -c.height)
    out, cum = [], 0.0
    for c in cohorts:
        p = pft_mod.get(c.pft)
        lai = c.biomass * leaf_fraction(p) * p.sla
        above = math.exp(-p.k_ext * min(cum, 12.0))
        below = math.exp(-p.k_ext * min(cum + lai, 12.0))
        out.append(max(0.0, above - below))
        cum += lai
    return out


def _pet_veg(t: Tile) -> float:
    from .hydrology import annual_pet

    return annual_pet(t.temp) * 1.1


def step_year(world: World) -> dict:
    """推进一年：生长、竞争、死亡、招募。"""
    total_npp = 0.0
    for t in world.tiles:
        if not t.cohort:
            t.lai = 0.0
            t.npp = 0.0
            continue
        pet_veg = _pet_veg(t)
        ordered = sorted(t.cohort, key=lambda c: -c.height)
        lights = light_profile(t)
        light_map = {id(c): l for c, l in zip(ordered, lights)}
        survivors: list[Cohort] = []
        npp_tile = 0.0
        for c in t.cohort:
            p = pft_mod.get(c.pft)
            fT = _temp_response(p, t.temp)
            fW = _water_response(p, t, pet_veg)
            light = light_map[id(c)]
            shade_penalty = math.exp(-(1.0 - p.shade_tol) * 2.0 * (1.0 - light))
            gpp = GPP_REF * (p.vcmax / 60.0) * fT * fW * min(1.0, light * 3.0) * shade_penalty
            npp = max(0.0, gpp * (1 - RESP_FRAC))
            npp_tile += npp

            grow = npp * 0.85
            c.biomass = min(BIOMASS_CAP, c.biomass + grow)
            if p.is_woody:
                c.height += (p.h_max - c.height) * 0.045 * max(0.2, fW)
            else:
                c.height = p.h_max * (0.7 + 0.3 * min(1.0, c.biomass / 0.6))
            c.age += 1

            stress = 1.0 - fW
            crowd = max(0.0, tile_lai(t) - 6.0) / 6.0
            mort = p.mortality * (1 + 2.5 * stress) + 0.18 * crowd
            c.biomass -= c.biomass * min(0.9, mort)
            if c.biomass > 1e-3:
                survivors.append(c)

        # 招募：种子库里有、气候合适、且当前空缺
        present = {c.pft for c in survivors}
        pool = world.meta.get("seed_pool", [p.key for p in pft_mod.LIBRARY.values()])
        for key in pool:
            p = pft_mod.get(key)
            if key in present:
                continue
            s = p.suitability(t.temp, t.precip)
            if s > 0.25 and tile_lai(t) < 4.5:
                survivors.append(Cohort(key, 0.02 * s, 0, p.h_max * 0.02))
                present.add(key)

        t.cohort = survivors
        t.lai = tile_lai(t)
        t.npp = npp_tile
        total_npp += npp_tile
    return {"mean_npp": total_npp / max(len(world.tiles), 1)}


def spin_up(world: World, years: int = 80) -> dict:
    """从初始队列推进到准平衡态，返回逐年轨迹（用于观察替换后的响应）。"""
    from . import hydrology

    traj = []
    for y in range(years):
        hydrology.step(world)
        info = step_year(world)
        bio = sum(t.biomass for t in world.tiles) / len(world.tiles)
        traj.append({"year": y + 1, "mean_npp": round(info["mean_npp"], 4),
                     "mean_biomass": round(bio, 4)})
    return {"years": years, "trajectory": traj}


def biome_counts(world: World) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in world.tiles:
        out[t.biome] = out.get(t.biome, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
