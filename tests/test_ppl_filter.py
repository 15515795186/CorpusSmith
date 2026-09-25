"""ppl_filter 单元测试
注意: 不加载真实模型 —— monkeypatch _ppl_of, 只测过滤/标定逻辑。
(模型加载路径由扫码脚本实测覆盖)
"""
import unittest

from file_processor.cleaners.ppl_filter import PPLFilter


class TestPPLFilterLogic(unittest.TestCase):
    def setUp(self):
        self.f = PPLFilter(threshold=None, quantile=0.90)
        # 假"打分器": 含 noise 的文本高 PPL, 正常文本低 PPL
        self.f._ppl_of = lambda text: 10.0 if "noise" in text else 3.0

    def test_scan_mode_keeps_all_and_collects(self):
        r1 = self.f.check("noise noise noise noise")
        r2 = self.f.check("this is a normal document body")
        self.assertTrue(r1.passed)
        self.assertTrue(r2.passed)
        self.assertEqual(self.f.scores, [10.0, 3.0])

    def test_threshold_mode_rejects_high_ppl(self):
        self.f.threshold = 5.0
        self.assertFalse(self.f.check("noise everywhere noise").passed)
        self.assertTrue(self.f.check("normal sentence here").passed)

    def test_short_text_always_passes(self):
        self.f.threshold = 1.0  # 即使阈值极小
        self.assertTrue(self.f.check("短").passed)
        self.assertEqual(self.f.scores, [])  # 不收集分数

    def test_calibrate_threshold_by_quantile(self):
        f = PPLFilter(quantile=0.90)
        f.scores = [float(i) for i in range(1, 11)]  # 1..10
        th = f.calibrate_threshold()
        self.assertEqual(th, 9.0)   # 90 分位数 = 第9个
        self.assertEqual(f.threshold, 9.0)

    def test_cache_dir_forwarded_to_from_pretrained(self):
        """模型缓存根必须显式透传, 不依赖外部环境变量(否则默认缓存里没模型时静默失败)"""
        from unittest import mock
        calls = {}

        def fake_tok(name, **kw):
            calls["tokenizer"] = (name, kw)
            return object()

        def fake_model(name, **kw):
            calls["model"] = (name, kw)
            return object()

        f = PPLFilter(model_name="Qwen/Qwen2.5-0.5B", cache_dir="D:/model")
        with mock.patch("transformers.AutoTokenizer.from_pretrained", side_effect=fake_tok), \
                mock.patch("transformers.AutoModelForCausalLM.from_pretrained",
                           side_effect=fake_model):
            f._ensure_model()
        self.assertIsNotNone(f._model)
        self.assertEqual(calls["tokenizer"][1].get("cache_dir"), "D:/model")
        self.assertEqual(calls["model"][1].get("cache_dir"), "D:/model")

    def test_model_load_failure_raises_with_actionable_hint(self):
        """模型加载失败必须 fail fast 且提示缓存根怎么配, 不得静默降级"""
        from unittest import mock
        f = PPLFilter(cache_dir="D:/definitely-missing-cache")
        with mock.patch("transformers.AutoTokenizer.from_pretrained",
                        side_effect=OSError("couldn't find them in the cached files")):
            with self.assertRaises(RuntimeError) as ctx:
                f._ensure_model()
        msg = str(ctx.exception)
        self.assertIn("cache_dir", msg)
        self.assertIn("definitely-missing-cache", msg)

    def test_calibrate_requires_scores(self):
        f = PPLFilter()
        with self.assertRaises(ValueError):
            f.calibrate_threshold()


if __name__ == "__main__":
    unittest.main()