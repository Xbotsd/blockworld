"""机器人在环测试：设备、链路、同化、判断策略、闭环收敛。"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blockworld import Criteria, materials
from blockworld import verifiers as V
from blockworld.robot import (Assimilator, Measurement, PolicyConfig, SimCPT,
                              SimLiDAR, Supervisor, make_device, make_transport)
from blockworld.robot.demo import make_scenario
from blockworld.site import build_from, synthesize
from blockworld.world import Region, WorldConfig


def small_world(seed: int = 11):
    return build_from(synthesize(WorldConfig(nx=10, ny=10, n_boreholes=5, seed=seed)))


class TestMeasurement(unittest.TestCase):
    def test_rejects_unknown_kind(self):
        with self.assertRaises(ValueError):
            Measurement(kind="temperature", x=0, y=0, value=1.0, sigma=0.1)

    def test_rejects_nonpositive_sigma(self):
        with self.assertRaises(ValueError):
            Measurement(kind="elevation", x=0, y=0, value=1.0, sigma=0.0)

    def test_round_trip(self):
        m = Measurement(kind="water_table", x=12.5, y=30.0, value=2.4,
                        sigma=0.2, device="probe")
        again = Measurement.from_dict(json.loads(json.dumps(m.to_dict())))
        self.assertEqual(again.kind, m.kind)
        self.assertAlmostEqual(again.value, m.value, places=4)


class TestDevices(unittest.TestCase):
    def test_lidar_deterministic(self):
        w = small_world()
        import numpy as np

        d = SimLiDAR(per_scan=4)
        a = [m.value for m in d.scan(w, np.random.default_rng(1))]
        b = [m.value for m in d.scan(w, np.random.default_rng(1))]
        self.assertEqual(a, b)

    def test_cpt_reports_material_present_at_depth(self):
        w = small_world()
        import numpy as np

        d = SimCPT(per_scan=3, max_depth=2.0)
        ms = d.scan(w, np.random.default_rng(2))
        self.assertTrue(ms)
        for m in ms:
            self.assertIn(m.kind, ("material_c", "material_phi"))
            self.assertIsNotNone(m.material)

    def test_registry_swap(self):
        from blockworld.robot import devices

        class Fake:
            name = "fake"

            def scan(self, truth, rng):
                return [Measurement("elevation", 0, 0, 1.0, 0.1, "fake")]

        devices.register_device("fake", Fake)
        try:
            dev = make_device("fake")
            self.assertEqual(dev.name, "fake")
            self.assertEqual(len(dev.scan(None, None)), 1)
        finally:
            devices.DEVICES.pop("fake", None)


class TestTransport(unittest.TestCase):
    def test_jsonl_write_then_replay(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.jsonl"
            w = make_transport(f"jsonl:{path}")
            w.open()
            w.send({"tick": 0, "type": "scan", "measurements": [1, 2]})
            w.send({"tick": 0, "type": "decision", "action": "rescan"})
            w.close()

            r = make_transport(f"replay:{path}")
            r.open()
            events = [r.recv() for _ in range(3)]
            r.close()
        self.assertEqual(len(events), 3)
        self.assertIsNone(events[-1])
        self.assertEqual(events[0]["type"], "scan")

    def test_unknown_transport_raises(self):
        with self.assertRaises(KeyError):
            make_transport("carrier-pigeon")


class TestAssimilation(unittest.TestCase):
    def test_elevation_updates_only_its_tile(self):
        w = small_world()
        t = w.tiles[0]
        neighbour = w.tiles[1]
        before = neighbour.elev
        a = Assimilator(w)
        rec = a.apply(Measurement("elevation", t.x, t.y, t.elev + 1.0, 0.05))
        self.assertTrue(rec.accepted)
        self.assertAlmostEqual(t.elev, t.elev, places=6)     # 已更新
        self.assertAlmostEqual(neighbour.elev, before, places=6)

    def test_outlier_is_rejected(self):
        w = small_world()
        t = w.tiles[0]
        a = Assimilator(w)
        rec = a.apply(Measurement("elevation", t.x, t.y, t.elev + 50.0, 0.05))
        self.assertFalse(rec.accepted)
        self.assertIn("异常", rec.reason)
        self.assertEqual(a.rejected, 1)

    def test_material_reclassification(self):
        w = small_world()
        soft_tiles = [t for t in w.tiles if any(m.key == "clay_soft" for _, m in t.col.layers)]
        if not soft_tiles:
            self.skipTest("本种子没有软黏土层")
        t = soft_tiles[0]
        idx = next(i for i, (_, m) in enumerate(t.col.layers) if m.key == "clay_soft")
        stiff = materials.get("clay_stiff")
        t.col.layers[idx] = (t.col.layers[idx][0], stiff)

        a = Assimilator(w)
        depth = sum(th for th, _ in t.col.layers[:idx]) + 0.5
        rec = a.apply(Measurement("material_c", t.x, t.y, 12.0, 3.0,
                                  depth=depth, material="clay_soft"))
        self.assertTrue(rec.accepted)
        keys = [m.key for _, m in t.col.layers]
        self.assertIn("clay_soft", keys)

    def test_absurd_water_table_is_rejected(self):
        w = small_world()
        t = w.tiles[0]
        a = Assimilator(w)
        rec = a.apply(Measurement("water_table", t.x, t.y, 999.0, 0.2))
        self.assertFalse(rec.accepted)
        self.assertEqual(a.rejected, 1)

    def test_water_table_clamped_when_gate_is_open(self):
        w = small_world()
        t = w.tiles[0]
        a = Assimilator(w, gate_sigma=1e9)
        rec = a.apply(Measurement("water_table", t.x, t.y, 999.0, 0.2))
        self.assertTrue(rec.accepted)
        self.assertLessEqual(t.wt, 30.0)


class TestPolicy(unittest.TestCase):
    def _verdict(self, margin: float, key: str = "infrastructure", passed: bool | None = None):
        return V.Verdict(key=key, title=key,
                         passed=(margin >= 0 if passed is None else passed),
                         margin=margin, metrics={}, reasons=[], inputs={})

    def test_rescan_when_evidence_short(self):
        sup = Supervisor()
        d = sup.decide([self._verdict(0.9)], coverage=0.9, evidence=3)
        self.assertEqual(d.action, "rescan")
        self.assertIn("证据不足", d.reason)

    def test_rescan_when_coverage_short(self):
        sup = Supervisor()
        d = sup.decide([self._verdict(0.9)], coverage=0.1, evidence=50)
        self.assertEqual(d.action, "rescan")

    def test_certify_when_confident(self):
        sup = Supervisor()
        d = sup.decide([self._verdict(0.9)], coverage=1.0, evidence=50)
        self.assertEqual(d.action, "certify")
        self.assertGreater(d.confidence, 0)

    def test_treat_when_failing_with_option(self):
        sup = Supervisor()
        d = sup.decide([self._verdict(-1.0)], coverage=1.0, evidence=50,
                       has_treatments=True)
        self.assertEqual(d.action, "treat")

    def test_abort_when_failing_without_option(self):
        sup = Supervisor()
        d = sup.decide([self._verdict(-1.0)], coverage=1.0, evidence=50,
                       has_treatments=False)
        self.assertEqual(d.action, "abort")

    def test_escalate_when_stagnant_below_certify_line(self):
        sup = Supervisor(PolicyConfig(certify_margin=0.5))
        d = sup.decide([self._verdict(0.2)], coverage=1.0, evidence=50, stagnant=True)
        self.assertEqual(d.action, "escalate")

    def test_p_fail_blocks_certification(self):
        sup = Supervisor()
        d = sup.decide([self._verdict(0.9)], coverage=1.0, evidence=50,
                       p_fail={"infrastructure": 0.8})
        self.assertIn(d.action, ("treat", "abort"))


class TestSession(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = make_scenario()
        cls.report = cls.sc.session.run()

    def test_loop_reacts_to_measurements(self):
        before = self.sc.session.before["infrastructure"]
        self.assertTrue(before["passed"], "先验应当是通过的")
        after_tick = self.report.ticks[0]["verdicts"][0]
        self.assertFalse(after_tick["passed"], "扫描后应当改判为不通过")

    def test_model_error_decreases(self):
        a = self.report.meta["model_error_before"]
        b = self.report.meta["model_error_at_decision"]
        self.assertLess(b["cohesion_mae_kpa"], a["cohesion_mae_kpa"])
        self.assertLess(b["material_mismatch"], a["material_mismatch"])

    def test_final_decision_is_actionable(self):
        self.assertIn(self.report.final.action, ("certify", "treat", "abort", "escalate"))

    def test_report_serializable(self):
        json.dumps(self.report.to_dict(), ensure_ascii=False)

    def test_acted_after_treat(self):
        actions = [t["decision"]["action"] for t in self.report.ticks]
        self.assertIn("treat", actions)
        self.assertTrue(self.report.meta["acted"])
        self.assertIn("certify", actions)

    def test_advice_only_mode_does_not_touch_model(self):
        sc = make_scenario(act=False, session_seed=4)
        rep = sc.session.run()
        self.assertFalse(rep.meta["acted"])
        self.assertEqual(rep.final.action, "treat")


if __name__ == "__main__":
    unittest.main(verbosity=2)
