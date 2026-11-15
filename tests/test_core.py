import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blockworld import animals, edits, geology, hydrology, materials, pft, sensitivity
from blockworld import verifiers as V
from blockworld.world import Cohort, Region, WorldConfig, build


def small_world(seed: int = 3):
    return build(WorldConfig(nx=16, ny=16, n_boreholes=8, seed=seed))


class TestBearingCapacity(unittest.TestCase):
    def test_strip_footing_matches_reference(self):
        # φ=30°、c=0、D=0 的条形基础：qu = 0.5·γ·B·Nγ，Nγ=22.4
        r = V.bearing_capacity(0.0, 30.0, 18.0, 1.0, D=0.0, shape="strip")
        self.assertAlmostEqual(r["qu_kpa"], 0.5 * 18.0 * 1.0 * 22.4, delta=2.0)

    def test_cohesion_dominates_at_zero_friction(self):
        r = V.bearing_capacity(20.0, 0.0, 18.0, 1.0, D=1.0, shape="square")
        self.assertAlmostEqual(r["factors"]["Nc"], 5.14, places=2)
        self.assertGreater(r["term_cohesion"], 0)

    def test_shape_and_depth_increase_capacity(self):
        strip = V.bearing_capacity(5, 30, 18, 2.0, D=1.0, shape="strip")["qu_kpa"]
        square = V.bearing_capacity(5, 30, 18, 2.0, D=1.0, shape="square")["qu_kpa"]
        deep = V.bearing_capacity(5, 30, 18, 2.0, D=3.0, shape="square")["qu_kpa"]
        self.assertGreater(square, strip)
        self.assertGreater(deep, square)


class TestMaterials(unittest.TestCase):
    def test_domain_rejects_bad_parameters(self):
        bad = materials.get("sand_dense").with_params(phi=60)
        errs, _ = bad.check()
        self.assertTrue(errs)

    def test_domain_rejects_wrong_constitutive(self):
        errs, _ = materials.get("gravel_dense").check("undrained")
        self.assertTrue(any("不支持" in e or "φu" in e for e in errs))

    def test_low_gravity_does_not_change_strength_parameters(self):
        m = materials.get("lunar_regolith")
        self.assertEqual(m.c, materials.get("lunar_regolith").c)
        self.assertGreater(m.gamma_sub, 0)


class TestGeology(unittest.TestCase):
    def test_krige_reproduces_samples(self):
        pts = np.array([[0, 0], [100, 0], [0, 100], [100, 100]], float)
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        out = geology.krige(pts, vals, pts)
        np.testing.assert_allclose(out, vals, atol=1e-4)

    def test_column_replace_preserves_thickness(self):
        col = geology.build_column(10.0, [(1.0, "silt"), (2.0, "clay_soft"), (3.0, "sand_dense")])
        before = col.thickness
        col.replace(0.5, 2.5, materials.get("gravel_dense"))
        self.assertAlmostEqual(col.thickness, before, places=6)
        self.assertEqual(col.material_at(1.0).key, "gravel_dense")
        self.assertEqual(col.material_at(0.2).key, "silt")

    def test_column_replace_out_of_range(self):
        col = geology.build_column(10.0, [(1.0, "silt")])
        with self.assertRaises(materials.DomainError):
            col.replace(0.0, 5.0, materials.get("sand_dense"))


class TestEdits(unittest.TestCase):
    def setUp(self):
        self.world = small_world()

    def test_rollback_restores_geology(self):
        t = self.world.tiles[0]
        before = [m.key for _, m in t.col.layers]
        s = edits.EditSession(self.world)
        s.set_material(Region.whole(), "gravel_dense", top_depth=0.0, bottom_depth=2.0)
        self.assertEqual(t.col.material_at(0.5).key, "gravel_dense")
        s.rollback(1)
        self.assertEqual([m.key for _, m in t.col.layers], before)

    def test_rejects_impossible_parameter_set(self):
        s = edits.EditSession(self.world)
        bad = materials.get("sand_dense").with_params(phi=70)
        with self.assertRaises(materials.DomainError):
            bad.validate()

    def test_rejects_pft_outside_climate_envelope(self):
        s = edits.EditSession(self.world)
        # 这个世界的年均温远低于热带雨林包络（12–34 °C，且降水 < 1400 mm）
        with self.assertRaises(pft.EnvelopeError):
            s.set_pft(Region.whole(), "tropical_rainforest", mode="replace")

    def test_gravity_edit_is_logged(self):
        s = edits.EditSession(self.world)
        s.set_gravity(1.62)
        self.assertAlmostEqual(self.world.config.gravity, 1.62, places=6)
        self.assertEqual(s.log()[-1]["op"], "set_gravity")


class TestVerifiers(unittest.TestCase):
    def setUp(self):
        self.world = small_world()
        hydrology.step(self.world)

    def test_bearing_capacity_fails_on_soft_clay(self):
        t = min(self.world.tiles, key=self.world.slope_deg)
        region = Region.circle(t.x, t.y, 80.0)
        edits.EditSession(self.world).set_material(
            region, "clay_soft", top_depth=0.0, bottom_depth=4.0)
        f = V.Foundation(B=1.5, L=1.5, D=1.0, load_kN=1500.0)
        v = V.verify_infrastructure(self.world, region, f)
        self.assertFalse(v.passed)
        self.assertLess(v.metrics["worst_fs"], 3.0)

    def test_landing_improves_with_cemented_surface(self):
        self.world.config.gravity = 1.62
        flat = min(self.world.tiles, key=self.world.slope_deg)
        region = Region.circle(flat.x, flat.y, 60.0)
        edits.EditSession(self.world).set_material(
            region, "lunar_regolith", top_depth=0.0, bottom_depth=2.0)
        lander = V.Lander(mass_kg=3000.0, n_pads=3, pad_diameter=0.25)
        before = V.verify_landing(self.world, region, lander)
        edits.EditSession(self.world).set_material(
            region, "cemented_regolith", top_depth=0.0, bottom_depth=2.0)
        after = V.verify_landing(self.world, region, lander)
        self.assertGreater(after.metrics["worst_fs"], before.metrics["worst_fs"])
        self.assertTrue(after.passed)

    def test_low_gravity_reduces_contact_pressure(self):
        flat = min(self.world.tiles, key=self.world.slope_deg)
        region = Region.circle(flat.x, flat.y, 60.0)
        lander = V.Lander(mass_kg=1000.0, n_pads=4, pad_diameter=0.5)
        self.world.config.gravity = 9.81
        earth = V.verify_landing(self.world, region, lander)
        self.world.config.gravity = 1.62
        moon = V.verify_landing(self.world, region, lander)
        self.assertLess(moon.metrics["contact_pressure_static_kpa"],
                        earth.metrics["contact_pressure_static_kpa"])


class TestEcology(unittest.TestCase):
    def test_forage_fraction_prefers_grassland(self):
        w = small_world(seed=5)
        t = w.tiles[0]
        t.cohort = [Cohort("temperate_broadleaf", 6.0, 50, 20.0)]
        forest = animals.forage_fraction(t)
        t.cohort = [Cohort("c4_grass", 0.8, 2, 1.5)]
        grass = animals.forage_fraction(t)
        self.assertGreater(grass, forest)

    def test_spin_up_produces_vegetation_and_biomes(self):
        w = small_world(seed=7)
        hydrology.step(w)
        from blockworld import vegetation

        vegetation.spin_up(w, years=12)
        self.assertGreater(sum(t.biomass for t in w.tiles), 0.0)
        self.assertTrue(all(t.biome for t in w.tiles))

    def test_large_herbivore_needs_more_energy(self):
        self.assertGreater(animals.fmr_kj_day(300.0), animals.fmr_kj_day(3.0))


class TestSensitivity(unittest.TestCase):
    def test_spearman_monotonic(self):
        x = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(sensitivity.spearman(x, [10, 20, 30, 40, 50]), 1.0, places=6)
        self.assertAlmostEqual(sensitivity.spearman(x, [5, 4, 3, 2, 1]), -1.0, places=6)

    def test_flip_detection(self):
        rows = [sensitivity.SweepRow(v, v - 2.0, v >= 3, v - 2.0) for v in [1, 2, 3, 4]]
        res = sensitivity.SweepResult("x", "X", rows)
        flips = res.flips
        self.assertEqual(len(flips), 1)
        self.assertEqual(flips[0]["direction"], "恢复")


class TestHydrology(unittest.TestCase):
    def test_pet_increases_with_temperature(self):
        self.assertGreater(hydrology.annual_pet(25.0), hydrology.annual_pet(5.0))

    def test_water_table_shallower_in_valley(self):
        w = small_world(seed=11)
        hydrology.step(w)
        low = [t for t in w.tiles if t.elev < np.percentile([x.elev for x in w.tiles], 20)]
        high = [t for t in w.tiles if t.elev > np.percentile([x.elev for x in w.tiles], 80)]
        self.assertLess(np.mean([t.wt for t in low]), np.mean([t.wt for t in high]) + 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
