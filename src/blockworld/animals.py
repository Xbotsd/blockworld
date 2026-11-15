"""动物侧推演：能量约束的个体代理模型（IBM）。

"区块上有没有动物"不是标签，而是能量收支的结果：个体按体重异速生长关系获得
取食率、代谢率与移动成本，在一个区块上摄取植被可供养的生物量，能量储备归零
即死亡。体重越大 → 绝对能量需求越高 → 只能在初级生产力足够高的区块维持正收支。

系数取 Kleiber 基础代谢与哺乳动物运动成本的经验量级，属于"可替换参数"，不是
精确的物种标定。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .world import World

KLEIBER = 293.0          # kJ/day per kg^0.75（70 kcal × 4.184）
FMR_FACTOR = 2.5          # 野外代谢率 / 基础代谢
DM_PER_C = 2.0            # 干物质 : 碳 ≈ 2
ENERGY_DM = 10_500.0      # kJ/kg 干物质（可消化能，而非燃烧热）
COT = 10.7                # 运动成本系数 J/(kg·m)，CoT = 10.7·m^-0.32
FORAGE_GRASS = 0.22       # 草地：可啃食的初级生产力比例（叶片占 NPP 的一大部分）
FORAGE_WOOD = 0.03        # 森林：绝大部分 NPP 变成木质，可食部分很低
FORAGE_BARE = 0.03


@dataclass(frozen=True)
class AnimalSpec:
    key: str
    name: str
    mass: float            # 成年体重 kg
    maturity: float        # 性成熟年龄 年
    lifespan: float        # 寿命 年
    diet: str = "herbivore"
    note: str = ""


SPECIES: dict[str, AnimalSpec] = {
    s.key: s
    for s in [
        AnimalSpec("vole", "小型啮齿类", 0.05, 0.4, 2.0, note="0.05 kg"),
        AnimalSpec("hare", "野兔", 3.0, 0.8, 5.0, note="3 kg"),
        AnimalSpec("deer", "中型鹿科", 60.0, 2.0, 12.0, note="60 kg"),
        AnimalSpec("bison", "大型牛科", 300.0, 4.0, 20.0, note="300 kg"),
    ]
}


@dataclass
class Animal:
    species: str
    mass: float
    tile: int
    reserve: float         # kJ
    age: float = 0.0
    alive: bool = True
    last_birth: float = -1.0


def fmr_kj_day(mass: float) -> float:
    return FMR_FACTOR * KLEIBER * mass**0.75


def forage_fraction(tile) -> float:
    """区块可供草食动物利用的初级生产力比例，由植被构成决定。

    草地叶片占 NPP 的比例高，森林绝大部分 NPP 变成木质，可食部分很低——这是
    "植被替换改变营养级容量"的具体耦合点。
    """
    if not tile.cohort:
        return FORAGE_BARE
    total = sum(c.biomass for c in tile.cohort) or 1.0
    frac = 0.0
    for c in tile.cohort:
        p = _pft(c.pft)
        frac += (c.biomass / total) * (FORAGE_WOOD if p.is_woody else FORAGE_GRASS)
    if any(_pft(c.pft).is_woody for c in tile.cohort):
        frac = max(frac, 0.035)      # 林下草本/灌木层
    return frac


def _pft(key: str):
    from . import pft as pft_mod

    return pft_mod.get(key)


def forage_kj_per_tile(tile, cell: float) -> float:
    """一个区块一年的可食能量（kJ）。"""
    kg_c = max(tile.npp, 0.0) * forage_fraction(tile) * cell**2
    return kg_c * DM_PER_C * ENERGY_DM


def home_range_tiles(mass: float, cell: float) -> int:
    """体重越大 → 家域越大。按 m^0.75 需求与面积的正比关系给出可达区块半径。"""
    area_ha = 0.05 * mass**0.9            # 经验量级：50 kg ≈ 1.7 ha，500 kg ≈ 14 ha
    return max(1, min(5, int(round((area_ha * 1e4 / cell**2) ** 0.5))))


def _move_cost_kj(mass: float, distance_m: float) -> float:
    return COT * mass ** (-0.32) * mass * distance_m / 1000.0


class Population:
    def __init__(self, world: World, seed: int = 7, initial: int = 240):
        self.world = world
        self.rng = np.random.default_rng(seed)
        self.animals: list[Animal] = []
        self.history: list[dict] = []
        self._spawn_initial(initial)

    # -- 初始化 ---------------------------------------------------------
    def _spawn_initial(self, n: int) -> None:
        keys = list(SPECIES)
        weights = np.array([1.0, 0.7, 0.35, 0.12])
        weights = weights / weights.sum()
        tiles = self.world.tiles
        for _ in range(n):
            sp = SPECIES[keys[int(self.rng.choice(len(keys), p=weights))]]
            t = tiles[int(self.rng.integers(0, len(tiles)))]
            reserve = fmr_kj_day(sp.mass) * 30.0 * float(self.rng.uniform(0.6, 1.4))
            self.animals.append(Animal(sp.key, sp.mass, self.world.idx(t.ix, t.iy), reserve,
                                       float(self.rng.uniform(0, sp.lifespan * 0.6))))

    # -- 推演 -----------------------------------------------------------
    def step(self, months: int = 1, dt_years: float = 1 / 12) -> dict:
        """推进 months 个月（每月一步）。"""
        for _ in range(months):
            self._step_month(dt_years)
        return self.stats()

    def _step_month(self, dt: float) -> None:
        world = self.world
        tiles = world.tiles
        cell = world.config.cell
        avail = np.array([forage_kj_per_tile(t, cell) / 12.0 for t in tiles])

        order = self.rng.permutation(len(self.animals))
        for oi in order:
            a = self.animals[int(oi)]
            if not a.alive:
                continue
            sp = SPECIES[a.species]
            need = fmr_kj_day(a.mass) * 30.0
            t = tiles[a.tile]

            take = min(need * 1.5, avail[a.tile] * 0.4)
            avail[a.tile] -= take
            a.reserve = min(a.reserve + take, need * 4.0)

            # 觅食移动：在自身体重决定的家域半径内采样候选区块
            if avail[a.tile] < need * 0.5:
                reach = home_range_tiles(a.mass, cell)
                best, best_score = a.tile, take
                for _ in range(8):
                    dx = int(self.rng.integers(-reach, reach + 1))
                    dy = int(self.rng.integers(-reach, reach + 1))
                    jx, jy = t.ix + dx, t.iy + dy
                    if not (0 <= jx < world.config.nx and 0 <= jy < world.config.ny):
                        continue
                    j = world.idx(jx, jy)
                    dist = cell * math.hypot(dx, dy)
                    gain = min(need, avail[j] * 0.25) - _move_cost_kj(a.mass, dist)
                    if gain > best_score:
                        best, best_score = j, gain
                if best != a.tile:
                    a.reserve -= _move_cost_kj(
                        a.mass,
                        cell * math.hypot(best % world.config.nx - t.ix,
                                          best // world.config.nx - t.iy),
                    )
                    a.tile = best

            a.reserve -= need * 1.0
            a.age += dt
            if a.mass < sp.mass:                       # 幼体生长
                a.mass = min(sp.mass, a.mass * 1.05)
            if a.reserve <= 0:
                a.alive = False
                continue
            if a.age > sp.lifespan:
                a.alive = False
                continue
            # 密度制约死亡（捕食、疾病、空间竞争）：局地密度越高越容易死
            local = len(tiles[a.tile].animals)
            p_die = 0.004 + min(0.30, 0.012 * local)
            if self.rng.random() < p_die:
                a.alive = False
                continue
            # 繁殖：能量储备超过两个月需求，且距离上次繁殖超过哺乳/育幼期
            juvenile = max(0.4, sp.maturity * 0.5)
            if (a.age > sp.maturity and a.reserve > need * 2.0
                    and a.age - a.last_birth > juvenile
                    and len(self.animals) < 3000 and self.rng.random() < 0.6):
                a.reserve -= need
                a.last_birth = a.age
                kid = Animal(a.species, a.mass * 0.6, a.tile, reserve=need * 0.4,
                             age=0.0, last_birth=-1.0)
                self.animals.append(kid)

        # 清理并记录
        self.animals = [a for a in self.animals if a.alive]
        self._flush_occupancy()

    def _flush_occupancy(self) -> None:
        for t in self.world.tiles:
            t.animals = []
        for i, a in enumerate(self.animals):
            self.world.tiles[a.tile].animals.append(i)

    # -- 统计 -----------------------------------------------------------
    def stats(self) -> dict:
        counts: dict[str, int] = {}
        for a in self.animals:
            counts[a.species] = counts.get(a.species, 0) + 1
        occupied = sum(1 for t in self.world.tiles if t.animals)
        return {
            "population": len(self.animals),
            "by_species": counts,
            "occupied_tiles": occupied,
            "occupancy": round(occupied / max(len(self.world.tiles), 1), 3),
        }

    def persist(self, species: str) -> bool:
        return any(a.species == species for a in self.animals)


def run(world: World, years: float = 5.0, seed: int = 7, initial: int = 240) -> dict:
    """在已平衡的植被场上运行动物推演，返回逐年统计。"""
    pop = Population(world, seed=seed, initial=initial)
    traj = []
    for y in range(int(round(years))):
        s = pop.step(months=12)
        s["year"] = y + 1
        traj.append(s)
    final = pop.stats()
    final["trajectory"] = traj
    final["surviving_species"] = sorted(final["by_species"])
    final["largest_survivor_kg"] = max(
        [SPECIES[k].mass for k in final["by_species"]], default=0.0
    )
    return final


def species_table() -> list[dict]:
    return [
        {
            "key": s.key,
            "name": s.name,
            "mass_kg": s.mass,
            "maturity_yr": s.maturity,
            "lifespan_yr": s.lifespan,
            "fmr_kj_day": round(fmr_kj_day(s.mass), 1),
        }
        for s in SPECIES.values()
    ]
