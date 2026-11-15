"""数据读写测试：往返一致、错误可诊断。"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blockworld import io
from blockworld.site import SiteSpec, synthesize
from blockworld.world import WorldConfig


def small_site():
    return synthesize(WorldConfig(nx=8, ny=8, n_boreholes=4, seed=5))


class TestSiteIO(unittest.TestCase):
    def test_round_trip(self):
        site = small_site()
        with tempfile.TemporaryDirectory() as d:
            path = io.save_site(site, Path(d) / "site.json")
            loaded = io.load_site(path)
        self.assertEqual(loaded.grid.nx, site.grid.nx)
        np.testing.assert_allclose(loaded.elevation, site.elevation, atol=1e-2)
        self.assertEqual(len(loaded.boreholes), len(site.boreholes))
        self.assertEqual(loaded.boreholes[0].layers[0][1],
                         site.boreholes[0].layers[0][1])
        self.assertAlmostEqual(loaded.gravity, site.gravity)

    def test_template_is_loadable(self):
        with tempfile.TemporaryDirectory() as d:
            files = io.write_site_template(d, nx=6, ny=5, cell=50.0)
            site = io.load_site(files[0])
        self.assertEqual((site.grid.nx, site.grid.ny), (6, 5))
        self.assertEqual(site.elevation.shape, (5, 6))
        self.assertGreater(len(site.boreholes), 0)
        self.assertIsNotNone(site.temp)
        self.assertIsNotNone(site.precip)

    def test_missing_file_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(io.SiteFormatError):
                io.load_site(Path(d) / "nope.json")

    def test_shape_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.savetxt(d / "dem.csv", np.zeros((4, 4)), delimiter=",")
            doc = {
                "grid": {"nx": 5, "ny": 5, "cell": 10},
                "elevation": {"csv": "dem.csv"},
                "boreholes": [{"x": 1, "y": 1, "ground": 10,
                               "layers": [[1.0, "silt"]]}],
            }
            (d / "site.json").write_text(__import__("json").dumps(doc), encoding="utf-8")
            with self.assertRaises(io.SiteFormatError):
                io.load_site(d / "site.json")

    def test_borehole_gap_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "bh.csv").write_text(
                "x,y,ground,top,bottom,material\n"
                "0,0,10,0,1,silt\n"
                "0,0,10,3,5,sand_dense\n", encoding="utf-8")
            with self.assertRaises(io.SiteFormatError):
                io.load_boreholes_csv(d / "bh.csv")

    def test_unknown_material_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "bh.csv").write_text(
                "x,y,ground,top,bottom,material\n"
                "0,0,10,0,1,unobtanium\n", encoding="utf-8")
            with self.assertRaises(io.SiteFormatError):
                io.load_boreholes_csv(d / "bh.csv")


if __name__ == "__main__":
    unittest.main(verbosity=2)
