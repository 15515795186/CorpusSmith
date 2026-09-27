# -*- coding: utf-8 -*-
"""provider 接线测试: 生成器 / 质量评分器按 llm_provider 选端点(全部离线, 不发请求)"""
import json
import os
import unittest

from file_processor.config import ProcessorConfig
from file_processor.llm.dashscope_generator import (
    DashScopeGenerator,
    OpenAICompatGenerator,
    PROVIDER_BASE_URLS,
    PROVIDER_ENV_KEYS,
)
from file_processor.quality_scorer import QualityScorer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LLMProviderTest(unittest.TestCase):

    def test_provider_tables(self):
        self.assertEqual(PROVIDER_BASE_URLS["deepseek"], "https://api.deepseek.com/")
        self.assertIn("dashscope", PROVIDER_BASE_URLS)
        self.assertEqual(PROVIDER_ENV_KEYS["deepseek"], "DEEPSEEK_API_KEY")
        self.assertEqual(PROVIDER_ENV_KEYS["dashscope"], "DASHSCOPE_API_KEY")

    def test_generator_uses_provider_endpoint_and_env_key(self):
        c = ProcessorConfig()
        g = DashScopeGenerator(api_key="k", model=c.llm_model(),
                               provider=c.llm_provider, base_url=c.llm_base_url())
        self.assertEqual(g.provider, "deepseek")
        self.assertEqual(g.base_url, "https://api.deepseek.com/")
        self.assertEqual(g.model, "deepseek-flash")
        self.assertIs(OpenAICompatGenerator, DashScopeGenerator)

        old = os.environ.get("DEEPSEEK_API_KEY")
        os.environ["DEEPSEEK_API_KEY"] = "env-key"
        try:
            g2 = DashScopeGenerator(model="deepseek-flash", provider="deepseek")
            self.assertEqual(g2.api_key, "env-key")
        finally:
            if old is None:
                os.environ.pop("DEEPSEEK_API_KEY", None)
            else:
                os.environ["DEEPSEEK_API_KEY"] = old

    def test_generator_omits_temperature_for_deepseek(self):
        """DeepSeek 端点对 temperature 敏感 → 只给非 deepseek 模型带上"""
        g = DashScopeGenerator(api_key="k", model="deepseek-flash",
                               provider="deepseek", base_url="https://api.deepseek.com/")
        kw = g._request_kwargs([{"role": "user", "content": "hi"}])
        self.assertNotIn("temperature", kw)
        g2 = DashScopeGenerator(api_key="k", model="qwen3.5-plus", provider="dashscope")
        self.assertIn("temperature", g2._request_kwargs([{"role": "user", "content": "hi"}]))

    def test_quality_scorer_uses_provider_endpoint(self):
        q = QualityScorer(api_key="k", model="deepseek-flash",
                          provider="deepseek", base_url="https://api.deepseek.com/")
        self.assertEqual(q.provider, "deepseek")
        self.assertEqual(q.base_url, "https://api.deepseek.com/")
        # 未显式给 base_url 时按 provider 兜底
        q2 = QualityScorer(api_key="k", model="deepseek-flash", provider="deepseek")
        self.assertEqual(q2.base_url, "https://api.deepseek.com/")
        q3 = QualityScorer(api_key="k", model="qwen-plus", provider="dashscope")
        self.assertTrue(q3.base_url.startswith("https://dashscope.aliyuncs.com/"))

    def test_run_configs_switched_to_deepseek(self):
        for rel in ("file_processor/run_config_v3.json",
                    "file_processor/run_config_cot_sft.json"):
            with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
                d = json.load(f)
            self.assertEqual(d.get("llm_provider"), "deepseek", rel)
            self.assertEqual(d.get("deepseek_model"), "deepseek-flash", rel)
            self.assertEqual(d.get("deepseek_base_url"), "https://api.deepseek.com/", rel)

    def test_cot_generator_wiring(self):
        """pipeline 的生成器按 provider 构造(不联网, 只验端点/模型)"""
        from file_processor.pipeline import FileProcessorPipeline
        c = ProcessorConfig(enable_llm=True, llm_provider="deepseek",
                            deepseek_api_key="k", output_dir=".")
        p = FileProcessorPipeline.__new__(FileProcessorPipeline)      # 绕开 __init__ 的重依赖, 只测生成器工厂
        p.config = c
        g = p._init_llm_generator()
        self.assertEqual(g.provider, "deepseek")
        self.assertEqual(g.base_url, "https://api.deepseek.com/")
        self.assertEqual(g.model, "deepseek-flash")


if __name__ == "__main__":
    unittest.main()