"""ppl_filter 单元测试
注意: 不加载真实模型 —— monkeypatch _ppl_of, 只测过滤/标定逻辑。
(模型加载路径由扫码脚本实测覆盖)

关于 transformers 的打桩方式:
    transformers / torch 已随本地 OCR 方案一并废弃, 不在 requirements.txt 与
    pyproject 依赖中。因此涉及 from_pretrained 的用例不能写成
    mock.patch("transformers.X") —— mock 解析字符串目标时会真实 import 该模块,
    在干净环境(含 CI)直接 ModuleNotFoundError。
    改为把假模块注入 sys.modules, 测试意图(缓存根透传 / 失败不静默)不变。
"""
import sys
import types
import unittest
from unittest import mock

from file_processor.cleaners.ppl_filter import PPLFilter


def _fake_transformers(tokenizer_side_effect=None, model_side_effect=None):
    """构造假 transformers 模块, 供 mock.patch.dict(sys.modules, ...) 注入。"""
    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = mock.MagicMock()
    fake.AutoModelForCausalLM = mock.MagicMock()
    if tokenizer_side_effect is not None:
        fake.AutoTokenizer.from_pretrained.side_effect = tokenizer_side_effect
    if model_side_effect is not None:
        fake.AutoModelForCausalLM.from_pretrained.side_effect = model_side_effect
    return fake


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
        calls = {}

        def fake_tok(name, **kw):
            calls["tokenizer"] = (name, kw)
            return object()

        def fake_model(name, **kw):
            calls["model"] = (name, kw)
            return object()

        f = PPLFilter(model_name="Qwen/Qwen2.5-0.5B", cache_dir="D:/model")
        fake = _fake_transformers(fake_tok, fake_model)
        with mock.patch.dict(sys.modules, {"transformers": fake}):
            f._ensure_model()
        self.assertIsNotNone(f._model)
        self.assertEqual(calls["tokenizer"][1].get("cache_dir"), "D:/model")
        self.assertEqual(calls["model"][1].get("cache_dir"), "D:/model")

    def test_model_load_failure_raises_with_actionable_hint(self):
        """模型加载失败必须 fail fast 且提示缓存根怎么配, 不得静默降级"""
        f = PPLFilter(cache_dir="D:/definitely-missing-cache")
        fake = _fake_transformers(
            tokenizer_side_effect=OSError("couldn't find them in the cached files"))
        with mock.patch.dict(sys.modules, {"transformers": fake}):
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
