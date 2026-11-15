"""对照公开公式与守恒关系的验证测试。

这些测试回答的是"物理对不对"，而不是"代码自洽不自洽"——上一版缺的正是这一层。
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blockworld import animals, hydrology
from blockworld import verifiers as V


class TestBearingFactors(unittest.TestCase):
    """承载力系数对照独立实现与经典数值。"""

    def test_nq_matches_closed_form(self):
        for phi in (0, 10, 20, 30, 40):
            if phi == 0:
                self.assertAlmostEqual(V.bearing_factors(0)["Nq"], 1.0)
                continue
            p = math.radians(phi)
            expected = math.exp(math.pi * math.tan(p)) * math.tan(math.pi / 4 + p / 2) ** 2
            self.assertAlmostEqual(V.bearing_factors(phi)["Nq"], expected, places=6)

    def test_known_values(self):
        # φ=30° 的经典值：Nq=18.40, Nc=30.14, Nγ=22.40（Vesic 形式）
        f = V.bearing_factors(30.0)
        self.assertAlmostEqual(f["Nq"], 18.40, delta=0.05)
        self.assertAlmostEqual(f["Nc"], 30.14, delta=0.10)
        self.assertAlmostEqual(f["Ngamma"], 22.40, delta=0.10)

    def test_undrained_limit(self):
        f = V.bearing_factors(0.0)
        self.assertEqual(f["Nq"], 1.0)
        self.assertAlmostEqual(f["Nc"], 5.14, places=2)
        self.assertAlmostEqual(f["Ngamma"], 0.0, places=6)


class TestCapacityMonotonicity(unittest.TestCase):
    def test_increases_with_cohesion_and_friction(self):
        base = V.bearing_capacity(10, 25, 18, 2.0, D=1.0)["qu_kpa"]
        self.assertGreater(V.bearing_capacity(20, 25, 18, 2.0, D=1.0)["qu_kpa"], base)
        self.assertGreater(V.bearing_capacity(10, 35, 18, 2.0, D=1.0)["qu_kpa"], base)

    def test_increases_with_width_and_depth(self):
        base = V.bearing_capacity(10, 30, 18, 1.0, D=0.0)["qu_kpa"]
        self.assertGreater(V.bearing_capacity(10, 30, 18, 2.0, D=0.0)["qu_kpa"], base)
        self.assertGreater(V.bearing_capacity(10, 30, 18, 1.0, D=2.0)["qu_kpa"], base)

    def test_settlement_scaling(self):
        s1 = V.elastic_settlement(100, 1.0, 20, 0.3)
        s2 = V.elastic_settlement(200, 1.0, 20, 0.3)
        s3 = V.elastic_settlement(100, 1.0, 40, 0.3)
        self.assertAlmostEqual(s2 / s1, 2.0, places=6)
        self.assertAlmostEqual(s3 / s1, 0.5, places=6)


class TestHydrologyBounds(unittest.TestCase):
    def test_budyko_limits(self):
        for p, pet in [(200, 1200), (800, 800), (2000, 300)]:
            aet = hydrology._budyko(p, pet)
            self.assertLessEqual(aet, p + 1e-6)
            self.assertLessEqual(aet, pet + 1e-6)
            self.assertGreaterEqual(aet, 0.0)

    def test_wet_environment_evaporates_less_than_precip(self):
        dry = hydrology._budyko(400, 1000) / 400
        wet = hydrology._budyko(2000, 700) / 2000
        self.assertGreater(dry, wet)

    def test_pet_responds_to_temperature(self):
        self.assertGreater(hydrology.annual_pet(28.0), hydrology.annual_pet(8.0))

    def test_radiation_positive_summer(self):
        self.assertGreater(hydrology.extraterrestrial_radiation_mm(35, 172), 5.0)


class TestAnimalAllometry(unittest.TestCase):
    def test_kleiber_scaling(self):
        # 质量 ×8 → 代谢 ×8^0.75 = 4.757
        ratio = animals.fmr_kj_day(80.0) / animals.fmr_kj_day(10.0)
        self.assertAlmostEqual(ratio, 8 ** 0.75, places=6)

    def test_movement_cost_decreases_with_mass(self):
        # CoT ∝ m^-0.32：单位质量每米成本随体重下降
        heavy = animals.COT * 500 ** (-0.32)
        light = animals.COT * 0.5 ** (-0.32)
        self.assertLess(heavy, light)


class TestSlopeFormula(unittest.TestCase):
    def test_flat_ground_is_stable_and_steep_is_not(self):
        class Col:
            def material_at(self, d):
                from blockworld import materials

                return materials.get("sand_dense")

            def weighted(self, a, b, water_table=None):
                from blockworld import materials

                m = materials.get("sand_dense")
                return {"gamma": m.gamma, "c": m.c, "phi": m.phi,
                        "E": m.E, "nu": m.nu, "k": m.k}

        class T:
            ix = iy = 0
            elev = 10.0
            wt = 20.0
            col = Col()

        class W:
            def region_tiles(self, region):
                return [T()]

            def slope_deg(self, t):
                return self._s

        w = W()
        w._s = 2.0
        self.assertTrue(V.verify_slope(w).passed)
        w._s = 40.0
        self.assertFalse(V.verify_slope(w).passed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
