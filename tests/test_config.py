import json
import os
import unittest

from file_processor.config import ProcessorConfig
from file_processor.dotenv import load_dotenv
from tests._tempdir import cleanup, workspace_tempdir


class DotenvTest(unittest.TestCase):
    """零依赖 .env 加载器：格式容忍、真实变量优先、缺失文件静默"""

    KEY = "DSH_DOTENV_TEST_KEY"

    def setUp(self):
        os.environ.pop(self.KEY, None)
        self.tmp = workspace_tempdir()
        self.addCleanup(lambda: cleanup(self.tmp))
        self.addCleanup(os.environ.pop, self.KEY, None)

    def _write_env(self, content):
        path = os.path.join(self.tmp, ".env")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_plain_comment_and_duplicate_key(self):
        path = self._write_env(
            f"# 注释行\n{self.KEY}=plain-value\n\n"
            f"{self.KEY}=ignored-second-occurrence\n")
        loaded = load_dotenv(path)
        self.assertEqual(os.environ[self.KEY], "plain-value",
                         "文件内重复键：先到先得，后者因变量已存在被跳过")
        self.assertIn(self.KEY, loaded)

    def test_export_prefix_and_single_quotes(self):
        path = self._write_env(f"export {self.KEY}='sk-quoted value'")
        load_dotenv(path)
        self.assertEqual(os.environ[self.KEY], "sk-quoted value")

    def test_real_env_wins_over_dotenv(self):
        os.environ[self.KEY] = "from-shell"
        path = self._write_env(f"{self.KEY}=from-file")
        load_dotenv(path)
        self.assertEqual(os.environ[self.KEY], "from-shell",
                         "shell 显式设置的变量不得被 .env 静默覆盖")

    def test_empty_env_is_filled(self):
        os.environ[self.KEY] = ""
        path = self._write_env(f"{self.KEY}=from-file")
        load_dotenv(path)
        self.assertEqual(os.environ[self.KEY], "from-file")

    def test_missing_file_is_silent(self):
        self.assertEqual(load_dotenv(os.path.join(self.tmp, "nope.env")), {})

    def test_default_path_points_to_project_root(self):
        from file_processor.dotenv import DEFAULT_ENV_PATH
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.assertEqual(os.path.dirname(DEFAULT_ENV_PATH), root)


class ConfigTest(unittest.TestCase):

    def test_default_input_dirs_point_to_data_dir(self):
        config = ProcessorConfig()
        self.assertTrue(config.input_dirs, "默认输入目录不应为空")
        for key, path in config.input_dirs.items():
            self.assertTrue(path.startswith("data/"), f"{key} 默认路径应为 data/ 下: {path}")
        self.assertEqual(config.output_dir, "datasets")

    def test_from_json_ignores_unknown_keys(self):
        tmp = workspace_tempdir()
        try:
            path = os.path.join(tmp, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"min_text_length": 55, "chunk_size": 256,
                           "this_key_does_not_exist": 1}, f)
            config = ProcessorConfig.from_json(path)
            self.assertEqual(config.min_text_length, 55)
            self.assertEqual(config.chunk_size, 256)
            self.assertFalse(hasattr(config, "this_key_does_not_exist"))
        finally:
            cleanup(tmp)

    def test_to_dict_redacts_api_keys(self):
        config = ProcessorConfig()
        config.dashscope_api_key = "sk-secret"
        config.deepseek_api_key = "sk-ds-secret"
        redacted = config.to_dict(redact=True)
        self.assertEqual(redacted["dashscope_api_key"], "***redacted***")
        self.assertEqual(redacted["deepseek_api_key"], "***redacted***")
        # 不 redact 时保留原值（供存储使用）
        self.assertEqual(config.to_dict()["dashscope_api_key"], "sk-secret")

    def test_api_key_prefers_config_over_env(self):
        """文本 LLM key 按 provider 解析: 默认 deepseek, 也可切回 dashscope"""
        config = ProcessorConfig()
        self.assertEqual(config.llm_provider, "deepseek")
        config.deepseek_api_key = "from-config"
        self.assertEqual(config.api_key(), "from-config")

        old = os.environ.get("DEEPSEEK_API_KEY")
        os.environ["DEEPSEEK_API_KEY"] = "from-env"
        try:
            empty = ProcessorConfig()
            self.assertEqual(empty.api_key(), "from-env")
        finally:
            if old is None:
                os.environ.pop("DEEPSEEK_API_KEY", None)
            else:
                os.environ["DEEPSEEK_API_KEY"] = old

        # 文本链 key 只认 deepseek; dashscope key 仅供 OCR 视觉链
        ocr = ProcessorConfig(dashscope_api_key="ds-config")
        self.assertEqual(ocr.ocr_api_key(), "ds-config")
        self.assertNotEqual(ocr.api_key(), "ds-config")

    def test_provider_resolution_and_ocr_key_separation(self):
        """模型/端点按 provider 解析; OCR(视觉) key 与文本 key 分离"""
        c = ProcessorConfig()
        self.assertEqual(c.llm_model(), "deepseek-flash")
        self.assertEqual(c.llm_base_url(), "https://api.deepseek.com/")
        self.assertEqual(c.quality_model_resolved(), "deepseek-flash")
        self.assertEqual(c.cot_model(), "deepseek-flash")

        old = os.environ.get("DASHSCOPE_API_KEY")
        old_ds = os.environ.get("DEEPSEEK_API_KEY")
        os.environ["DASHSCOPE_API_KEY"] = "ocr-only"
        os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            self.assertEqual(c.ocr_api_key(), "ocr-only")
            # 只有 OCR key 时, 文本 LLM 视为未配置(不误开质量评分)
            self.assertEqual(ProcessorConfig().api_key(), "")
        finally:
            if old is None:
                os.environ.pop("DASHSCOPE_API_KEY", None)
            else:
                os.environ["DASHSCOPE_API_KEY"] = old
            if old_ds is not None:
                os.environ["DEEPSEEK_API_KEY"] = old_ds

        # 文本链已全量切 DeepSeek: dashscope 不参与文本生成(仅 OCR 保留其 key)
        e = ProcessorConfig(llm_cot_model="deepseek-v4-pro")
        self.assertEqual(e.llm_model(), "deepseek-flash")
        self.assertEqual(e.cot_model(), "deepseek-v4-pro")
        self.assertEqual(e.llm_base_url(), "https://api.deepseek.com/")
        self.assertFalse(hasattr(e, "dashscope_model"))

    def test_repo_run_configs_are_valid(self):
        """仓库内所有 run_config*.json 都能解析, 且关键项合法"""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_dir = os.path.join(base, "file_processor")
        names = sorted(f for f in os.listdir(cfg_dir)
                       if f.startswith("run_config") and f.endswith(".json"))
        self.assertTrue(names, "至少应存在一个运行配置")
        # 当前保留的两个入口: 全流程 / 只跑 COT
        self.assertIn("run_config_v3.json", names)
        self.assertIn("run_config_cot_sft.json", names)
        for name in names:
            config = ProcessorConfig.from_json(os.path.join(cfg_dir, name))
            # 各配置的 output_types 不同(cot_sft 只出 cot), 只校验合法与非空
            self.assertTrue(config.output_types, name)
            self.assertTrue(set(config.output_types) <= {"pretrain", "instruction", "cot"},
                            name)
            self.assertGreater(config.max_workers, 0)


if __name__ == "__main__":
    unittest.main()