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
        config.openai_api_key = "sk-secret-2"
        redacted = config.to_dict(redact=True)
        self.assertEqual(redacted["dashscope_api_key"], "***redacted***")
        self.assertEqual(redacted["openai_api_key"], "***redacted***")
        # 不 redact 时保留原值（供存储使用）
        self.assertEqual(config.to_dict()["dashscope_api_key"], "sk-secret")

    def test_api_key_prefers_config_over_env(self):
        config = ProcessorConfig()
        config.dashscope_api_key = "from-config"
        self.assertEqual(config.api_key(), "from-config")

        old = os.environ.get("DASHSCOPE_API_KEY")
        os.environ["DASHSCOPE_API_KEY"] = "from-env"
        try:
            empty = ProcessorConfig()
            self.assertEqual(empty.api_key(), "from-env")
        finally:
            if old is None:
                os.environ.pop("DASHSCOPE_API_KEY", None)
            else:
                os.environ["DASHSCOPE_API_KEY"] = old

    def test_repo_run_configs_are_valid(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in ("run_config.json", "run_config_ocr.json"):
            path = os.path.join(base, "file_processor", name)
            config = ProcessorConfig.from_json(path)
            self.assertIn("pretrain", config.output_types)
            self.assertGreater(config.max_workers, 0)


if __name__ == "__main__":
    unittest.main()