# -*- coding: utf-8 -*-
"""CLI 开关语义: --resume / --rebuild 的优先级

要点: 配置里写了 resume=true(如 run_config_v3.json)时, 必须能用 --rebuild 显式覆盖,
否则"清空重建"就没有命令行出口。
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from file_processor import main as fp_main                                        # noqa: E402


def _config(*argv):
    with mock.patch.object(sys, "argv", ["file_processor.main"] + list(argv)):
        return fp_main.build_config(fp_main.parse_args())


class ResumeRebuildFlagTest(unittest.TestCase):

    def test_default_is_production_resume(self):
        """默认参数 = v3 生产配置: resume=true(只补不覆盖), 输出 = 最终产出目录"""
        cfg = _config()
        self.assertTrue(cfg.resume)
        self.assertEqual(cfg.output_dir, "datasets_v3_20260927")

    def test_resume_flag(self):
        self.assertTrue(_config("--resume").resume)

    def test_config_can_enable_resume(self):
        """run_config_v3.json 里 resume=true → 配置生效"""
        self.assertTrue(_config("--config", "file_processor/run_config_v3.json").resume)

    def test_rebuild_overrides_config_resume(self):
        """--rebuild 必须能覆盖配置里的 resume=true"""
        cfg = _config("--config", "file_processor/run_config_v3.json", "--rebuild")
        self.assertFalse(cfg.resume)

    def test_rebuild_wins_over_resume(self):
        self.assertFalse(_config("--resume", "--rebuild").resume)

    def test_rebuild_does_not_touch_output_dir(self):
        """--rebuild 只切换续跑语义, 不改输出目录"""
        cfg = _config("--config", "file_processor/run_config_v3.json", "--rebuild")
        self.assertEqual(cfg.output_dir, "datasets_v3_20260927")


if __name__ == "__main__":
    unittest.main()