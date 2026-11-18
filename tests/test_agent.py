"""离线顾问测试：解析鲁棒性、回退、矛盾检测、请求体正确性（全程不联网）。"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blockworld.agent import (Advice, Advisor, Facts, NullClient, OllamaClient,
                              OpenAICompatClient, ScriptedClient, make_client,
                              parse_advice, render_brief)
from blockworld.agent.llm import CLIENTS


class FakeFacts(Facts):
    pass


def facts(**over) -> Facts:
    base = Facts(
        scenario={"prior": "clay_soft→clay_stiff", "design": "1.5×1.5 m, 400 kN"},
        prior={"infrastructure": {"passed": True, "margin": 0.5}},
        timeline=[
            {"tick": 0, "coverage": 0.44, "evidence": 11, "accepted": 11,
             "rejected": 0, "margins": {"infrastructure": -1.54},
             "action": "rescan", "reason": "证据不足", "treatment": None,
             "applied": False},
            {"tick": 1, "coverage": 1.0, "evidence": 13, "accepted": 13,
             "rejected": 0, "margins": {"infrastructure": -1.54},
             "action": "treat", "reason": "未通过", "treatment": "换填 6 m 砂砾",
             "applied": True},
        ],
        final={"action": "certify", "reason": "全部通过", "confidence": 0.6,
               "payload": {}},
        evidence={"measurements": 70, "assimilated": 66, "rejected": 4,
                  "coverage": 1.0, "scope_tiles": 9, "measured_tiles": 9},
        model_error={"before": {"cohesion_mae_kpa": 16.5, "phi_mae_deg": 4.0,
                                "material_mismatch": 0.5},
                     "after": None,
                     "at_decision": {"cohesion_mae_kpa": 2.67, "phi_mae_deg": 0.62,
                                     "material_mismatch": 0.056}},
        policy={"certify_margin": 0.3, "min_coverage": 0.9},
    )
    for k, v in over.items():
        setattr(base, k, v)
    return base


GOOD = json.dumps({
    "summary": "先验误判，扫描纠正后换填并通过。",
    "verdict": "certify",
    "risks": ["未测区块仍依赖先验"],
    "recommendations": ["对基础范围补 3 个 CPT 点"],
    "caveats": ["结论依赖 4σ 门控没有误杀"],
    "confidence": 0.7,
}, ensure_ascii=False)


class TestParsing(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_advice(GOOD)["verdict"], "certify")

    def test_fenced_json(self):
        text = "这是结果：\n```json\n" + GOOD + "\n```\n以上。"
        self.assertEqual(parse_advice(text)["verdict"], "certify")

    def test_surrounded_by_prose(self):
        self.assertEqual(parse_advice("好的，" + GOOD + " 完毕")["verdict"], "certify")

    def test_garbage_returns_none(self):
        self.assertIsNone(parse_advice("我不知道该怎么回答。"))
        self.assertIsNone(parse_advice(""))


class TestAdvisor(unittest.TestCase):
    def test_llm_path(self):
        client = ScriptedClient([GOOD])
        adv = Advisor(client).advise(facts())
        self.assertEqual(adv.source, "llm:scripted/scripted")
        self.assertEqual(adv.verdict, "certify")
        self.assertTrue(adv.agrees_with_policy)
        self.assertTrue(adv.advisory_only)
        self.assertEqual(len(client.calls), 1)
        prompt = json.dumps(client.calls[0], ensure_ascii=False)
        self.assertIn("换填 6 m 砂砾", prompt)
        self.assertIn("16.5", prompt)

    def test_disagreement_is_recorded_not_applied(self):
        bad = json.loads(GOOD)
        bad["verdict"] = "abort"
        adv = Advisor(ScriptedClient([json.dumps(bad, ensure_ascii=False)])).advise(facts())
        self.assertFalse(adv.agrees_with_policy)
        self.assertTrue(any("以策略为准" in c for c in adv.caveats))
        self.assertEqual(adv.verdict, "abort")      # 记录模型意见
        # 但事实包里的判定没有被改写
        self.assertEqual(facts().final["action"], "certify")

    def test_invalid_json_falls_back(self):
        adv = Advisor(ScriptedClient(["对不起，我无法回答"])).advise(facts())
        self.assertTrue(adv.source.startswith("fallback"))
        self.assertEqual(adv.verdict, "certify")

    def test_unavailable_client_falls_back(self):
        adv = Advisor(NullClient()).advise(facts())
        self.assertTrue(adv.source.startswith("fallback"))
        self.assertIn("certify", adv.summary)

    def test_no_response_falls_back(self):
        adv = Advisor(ScriptedClient([], up=True)).advise(facts())
        self.assertTrue(adv.source.startswith("fallback"))

    def test_confidence_is_clamped(self):
        d = json.loads(GOOD)
        d["confidence"] = 7
        adv = Advisor(ScriptedClient([json.dumps(d)])).advise(facts())
        self.assertEqual(adv.confidence, 1.0)

    def test_oversized_facts_are_truncated(self):
        f = facts()
        f.timeline = f.timeline * 400
        client = ScriptedClient([GOOD])
        Advisor(client, max_facts_chars=500).advise(f)
        user = client.calls[0][1]["content"]
        self.assertIn("截断", user)


class TestClients(unittest.TestCase):
    def test_ollama_payload(self):
        seen = {}

        def fake_post(url, payload, headers, timeout):
            seen.update({"url": url, "payload": payload, "headers": headers})
            return {"message": {"content": GOOD}}

        c = OllamaClient(model="qwen2.5:7b", post=fake_post)
        out = c.chat([{"role": "user", "content": "hi"}], json_mode=True)
        self.assertEqual(out, GOOD)
        self.assertTrue(seen["url"].endswith("/api/chat"))
        self.assertEqual(seen["payload"]["model"], "qwen2.5:7b")
        self.assertFalse(seen["payload"]["stream"])
        self.assertEqual(seen["payload"]["format"], "json")

    def test_ollama_unavailable_is_false(self):
        def boom(*a, **k):
            raise OSError("connection refused")

        c = OllamaClient(get=boom)
        self.assertFalse(c.available())
        self.assertIsNone(c.chat([{"role": "user", "content": "x"}]))

    def test_openai_compat_payload(self):
        seen = {}

        def fake_post(url, payload, headers, timeout):
            seen.update({"url": url, "auth": headers.get("Authorization")})
            return {"choices": [{"message": {"content": GOOD}}]}

        c = OpenAICompatClient(model="local", post=fake_post)
        self.assertEqual(c.chat([{"role": "user", "content": "x"}]), GOOD)
        self.assertTrue(seen["url"].endswith("/chat/completions"))
        self.assertTrue(seen["auth"].startswith("Bearer "))

    def test_registry(self):
        self.assertIn("ollama", CLIENTS)
        self.assertIsInstance(make_client("none"), NullClient)
        with self.assertRaises(KeyError):
            make_client("gpt-in-the-cloud")


class TestBrief(unittest.TestCase):
    def test_brief_contains_key_sections(self):
        f = facts()
        adv = Advisor(ScriptedClient([GOOD])).advise(f)
        md = render_brief(f, adv)
        for token in ("# 现场简报", "## 结论", "## 判据", "## 闭环时间线",
                      "## 风险", "## 声明", "换填 6 m 砂砾"):
            self.assertIn(token, md)

    def test_brief_marks_disagreement(self):
        bad = json.loads(GOOD)
        bad["verdict"] = "abort"
        f = facts()
        adv = Advisor(ScriptedClient([json.dumps(bad, ensure_ascii=False)])).advise(f)
        md = render_brief(f, adv)
        self.assertIn("以策略为准", md)

    def test_brief_works_on_fallback(self):
        f = facts()
        md = render_brief(f, Advisor(NullClient()).advise(f))
        self.assertIn("确定性摘要替代", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
