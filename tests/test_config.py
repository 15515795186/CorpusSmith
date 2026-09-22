import json
import os
import unittest

from file_processor.config import ProcessorConfig
from tests._tempdir import cleanup, workspace_tempdir


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