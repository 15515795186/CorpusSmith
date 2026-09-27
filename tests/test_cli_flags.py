# -*- coding: utf-8 -*-
"""CLI 开关语义: --resume / --rebuild 的优先级

要点: 默认(生产参数)就是 resume=true, 所以必须能用 --rebuild 显式切回"清空重建",
否则就没有命令行出口。
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
        """默认参数 = 最终生产配置: resume=true(只补不覆盖), 输出 = 最终产出目录"""
        cfg = _config()
        self.assertTrue(cfg.resume)
        self.assertEqual(cfg.output_dir, "datasets")

    def test_resume_flag(self):
        self.assertTrue(_config("--resume").resume)

    def test_cot_config_resume(self):
        """COT 专用配置自带 resume=false, 也会被 CLI 覆盖逻辑正确处理"""
        cfg = _config("--config", "file_processor/run_config_cot_sft.json")
        self.assertEqual(cfg.output_types, ["cot"])

    def test_rebuild_overrides_default_resume(self):
        """--rebuild 必须能覆盖默认/配置里的 resume=true"""
        self.assertFalse(_config("--rebuild").resume)
        cfg = _config("--config", "file_processor/run_config_cot_sft.json", "--rebuild")
        self.assertFalse(cfg.resume)

    def test_rebuild_wins_over_resume(self):
        self.assertFalse(_config("--resume", "--rebuild").resume)

    def test_rebuild_does_not_touch_output_dir(self):
        """--rebuild 只切换续跑语义, 不改输出目录"""
        self.assertEqual(_config("--rebuild").output_dir, "datasets")


if __name__ == "__main__":
    unittest.main()
