"""公开 API 的契约测试：可序列化、可回滚、不静默失败。"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blockworld import Criteria, Intervention, Study, build_from, synthesize
from blockworld.world import WorldConfig


def small_site(seed: int = 21):
    return synthesize(WorldConfig(nx=12, ny=12, n_boreholes=6, seed=seed))


class TestSite(unittest.TestCase):
    def test_build_from_requires_dem_and_boreholes(self):
        site = small_site()
        site.elevation = None
        with self.assertRaises(ValueError):
            build_from(site)

    def test_build_from_rejects_wrong_shape(self):
        site = small_site()
        site.temp = site.temp[:-1]
        with self.assertRaises(ValueError):
            build_from(site)

    def test_provenance_is_carried_into_world(self):
        site = small_site()
        world = build_from(site)
        self.assertEqual(world.meta["source"], "synthetic")

    def test_synthetic_is_deterministic(self):
        a = build_from(small_site(3))
        b = build_from(small_site(3))
        self.assertEqual([t.elev for t in a.tiles][:20], [t.elev for t in b.tiles][:20])


class TestIntervention(unittest.TestCase):
    def test_round_trip(self):
        iv = Intervention("填方", [{"op": "elevation", "delta": 2.0,
                                    "region": {"kind": "circle", "cx": 100,
                                               "cy": 100, "r": 50}}])
        again = Intervention.from_dict(json.loads(json.dumps(iv.to_dict())))
        self.assertEqual(again.key, iv.key)
        self.assertEqual(again.ops[0]["op"], "elevation")

    def test_invalid_material_raises(self):
        site = small_site()
        world = build_from(site)
        iv = Intervention("坏替换", [{"op": "material", "material": "unobtanium",
                                      "top_depth": 0, "bottom_depth": 1}])
        with self.assertRaises(Exception):
            iv.apply(world)

    def test_unknown_op_raises(self):
        world = build_from(small_site())
        iv = Intervention("坏操作", [{"op": "teleport"}])
        with self.assertRaises(ValueError):
            iv.apply(world)


class TestStudy(unittest.TestCase):
    def setUp(self):
        self.site = small_site()
        self.criteria = Criteria(purposes=("infrastructure", "landing"))

    def test_run_returns_serializable_result(self):
        study = Study(self.site, self.criteria, interventions=(
            Intervention("换填", [{"op": "material", "material": "gravel_dense",
                                   "top_depth": 0.0, "bottom_depth": 6.0}]),))
        result = study.run()
        payload = result.to_dict()
        json.dumps(payload, ensure_ascii=False)          # 不抛异常即可
        passed = result.passed()
        self.assertIn("baseline", passed)
        self.assertIn("换填", passed)
        self.assertEqual(len(result.baseline), 2)

    def test_verdicts_have_margin(self):
        result = Study(self.site, self.criteria).run()
        for v in result.baseline:
            self.assertIn("margin", v)
            self.assertIn("passed", v)
            self.assertIn("reasons", v)

    def test_improvement_shows_positive_margin_delta(self):
        study = Study(self.site, self.criteria, interventions=(
            Intervention("换填", [{"op": "material", "material": "gravel_dense",
                                   "top_depth": 0.0, "bottom_depth": 6.0}]),))
        result = study.run()
        infra = [d for d in result.interventions["换填"]["deltas"]
                 if d["criterion"] == "infrastructure"][0]
        self.assertGreater(infra["margin_delta"], 0)

    def test_site_selection_returns_regions_inside_world(self):
        world = build_from(self.site)
        from blockworld.api import choose_foundation_site, choose_landing_site

        for region in (choose_foundation_site(world), choose_landing_site(world)):
            self.assertTrue(world.region_tiles(region))


if __name__ == "__main__":
    unittest.main(verbosity=2)
