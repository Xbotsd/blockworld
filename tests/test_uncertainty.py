"""不确定性模块测试：分布、可复现性、退化情形、Morris 排序。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blockworld import Criteria, Distribution, Intervention, Parameter, Study
from blockworld import uncertainty
from blockworld.site import synthesize
from blockworld.world import WorldConfig


def small_site(seed: int = 31):
    return synthesize(WorldConfig(nx=10, ny=10, n_boreholes=5, seed=seed))


def small_study():
    return Study(
        site=small_site(),
        criteria=Criteria(purposes=("infrastructure", "landing")),
        interventions=(Intervention("换填", [
            {"op": "material", "material": "gravel_dense",
             "top_depth": 0.0, "bottom_depth": 6.0}]),),
    )


class TestDistribution(unittest.TestCase):
    def test_uniform_bounds(self):
        d = Distribution("uniform", 2.0, 5.0)
        rng = np.random.default_rng(0)
        draws = [d.sample(rng) for _ in range(200)]
        self.assertGreaterEqual(min(draws), 2.0)
        self.assertLessEqual(max(draws), 5.0)

    def test_unit_to_value_endpoints(self):
        d = Distribution("uniform", -1.0, 3.0)
        self.assertAlmostEqual(d.unit_to_value(0.0), -1.0)
        self.assertAlmostEqual(d.unit_to_value(1.0), 3.0)

    def test_triangular_mode_validation(self):
        with self.assertRaises(ValueError):
            Distribution("triangular", 0.0, 1.0, mode=2.0)

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            Distribution("cauchy", 0, 1)


class TestMonteCarlo(unittest.TestCase):
    def test_deterministic_with_same_seed(self):
        study = small_study()
        p = Parameter("pad", {"kind": "lander", "field": "pad_diameter"},
                      Distribution("uniform", 0.2, 0.9))
        a = uncertainty.monte_carlo(study, [p], n=12, seed=7)
        b = uncertainty.monte_carlo(study, [p], n=12, seed=7)
        self.assertEqual(a.baseline["landing"]["p_fail"], b.baseline["landing"]["p_fail"])
        self.assertEqual(a.samples, b.samples)

    def test_degenerate_always_pass(self):
        study = small_study()
        # 足垫足够大 → 着陆判据恒通过
        p = Parameter("pad", {"kind": "lander", "field": "pad_diameter"},
                      Distribution("uniform", 5.0, 6.0))
        res = uncertainty.monte_carlo(study, [p], n=8, seed=1)
        self.assertEqual(res.baseline["landing"]["p_fail"], 0.0)

    def test_degenerate_always_fail(self):
        study = small_study()
        # 足垫极小 → 接触压力极大 → 恒失败
        p = Parameter("pad", {"kind": "lander", "field": "pad_diameter"},
                      Distribution("uniform", 0.01, 0.02))
        res = uncertainty.monte_carlo(study, [p], n=8, seed=1)
        self.assertEqual(res.baseline["landing"]["p_fail"], 1.0)

    def test_material_parameter_applies(self):
        study = small_study()
        p = Parameter("c", {"kind": "material_prop", "material": "clay_soft",
                            "prop": "c"}, Distribution("uniform", 5.0, 60.0))
        res = uncertainty.monte_carlo(study, [p], n=6, seed=2)
        self.assertIn("infrastructure", res.baseline)
        self.assertTrue(0.0 <= res.baseline["infrastructure"]["p_fail"] <= 1.0)

    def test_rejects_tiny_n(self):
        with self.assertRaises(ValueError):
            uncertainty.monte_carlo(small_study(), [], n=1)


class TestMorris(unittest.TestCase):
    def test_budget_guard(self):
        study = small_study()
        params = [
            Parameter(f"p{i}", {"kind": "lander", "field": "pad_diameter"},
                      Distribution("uniform", 0.2, 0.9))
            for i in range(12)
        ]
        with self.assertRaises(ValueError):
            uncertainty.morris(study, params, r=10, max_evaluations=50)

    def test_ranks_influential_parameter_first(self):
        study = small_study()
        pad = Parameter("pad", {"kind": "lander", "field": "pad_diameter"},
                        Distribution("uniform", 0.15, 0.9))
        # 一个几乎不起作用的参数：改变足垫质量在承载判据里影响很小
        mass = Parameter("mass", {"kind": "lander", "field": "mass_kg"},
                         Distribution("uniform", 3000.0, 3010.0))
        res = uncertainty.morris(study, [pad, mass], r=4, seed=3,
                                 intervention=study.interventions[0])
        ranking = res.ranking("landing", use="intervention")
        self.assertEqual(ranking[0]["param"], "pad")
        self.assertGreater(ranking[0]["mu_star"], ranking[-1]["mu_star"])

    def test_evaluation_count(self):
        study = small_study()
        params = [Parameter("pad", {"kind": "lander", "field": "pad_diameter"},
                            Distribution("uniform", 0.2, 0.9))]
        res = uncertainty.morris(study, params, r=3)
        self.assertEqual(res.evaluations, 3 * (1 + 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
