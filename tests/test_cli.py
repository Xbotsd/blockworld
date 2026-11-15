"""CLI 契约测试：帮助文本、模板生成、研究流程、错误退出码。"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blockworld import cli


class TestCli(unittest.TestCase):
    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_no_command_prints_help(self):
        self.assertEqual(cli.main([]), 1)

    def test_template_then_study(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.assertEqual(cli.main(["template", str(d / "site"),
                                       "--nx", "6", "--ny", "6", "--cell", "100"]), 0)
            cfg = d / "study.json"
            cfg.write_text(
                '{"site": "site/site.json", "seed": 1,'
                ' "criteria": {"purposes": ["infrastructure"]},'
                ' "interventions": [{"key": "fill", "ops": ['
                '  {"op": "material", "material": "gravel_dense",'
                '   "top_depth": 0, "bottom_depth": 4,'
                '   "region": {"kind": "all"}}]}]}',
                encoding="utf-8",
            )
            out = d / "res"
            self.assertEqual(cli.main(["study", str(cfg), "--out", str(out)]), 0)
            self.assertTrue((out / "study.json").exists())
            text = (out / "study.json").read_text(encoding="utf-8")
            self.assertIn("baseline", text)
            self.assertIn("fill", text)

    def test_study_reports_bad_config(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "bad.json"
            bad.write_text('{"site": "missing.json"}', encoding="utf-8")
            with self.assertRaises(Exception):
                cli.main(["study", str(bad), "--out", str(Path(d) / "o")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
