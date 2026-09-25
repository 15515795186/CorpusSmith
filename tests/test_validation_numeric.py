"""utils.validation 数字一致性回归测试。
覆盖: 行首编号误杀修复(cfg19 真 bug)、full_material 节选兜底、原幻觉拦截不放松。
"""
import unittest

from file_processor.utils.validation import (
    numeric_consistency,
    numbers_in,
    strip_bullet_numbers,
)


class TestStripBulletNumbers(unittest.TestCase):
    def test_strips_line_leading_numbers(self):
        self.assertEqual(strip_bullet_numbers("1. 内容\n2. 内容"), "内容\n内容")
        self.assertEqual(strip_bullet_numbers("(2) 内容"), "内容")
        self.assertEqual(strip_bullet_numbers("4、内容"), "内容")

    def test_keeps_decimal_and_body_numbers(self):
        # "2.5%" 不是行首编号, 不能剥掉"2."
        self.assertIn("2.5%", strip_bullet_numbers("16.9% 涨幅, 2.5% 降幅"))
        self.assertEqual(strip_bullet_numbers(""), "")


class TestNumericConsistency(unittest.TestCase):
    MAT = "2024年GDP能耗下降2.5%, 2025年底非化石能源占比达到18.9%, 碳排放强度下降3.9%。"

    def test_bullet_list_not_false_positive(self):
        # cfg19 真 bug: 要点化列举被行首编号数字误杀
        out = ("1. 2024 年 GDP 能耗降 2.5%\n"
               "2. 碳排放强度降 3.9%\n"
               "3. 2025 年底非化石占比 18.9%")
        self.assertTrue(numeric_consistency(out, self.MAT))

    def test_true_hallucination_still_rejected(self):
        out = "1. 第一行\n2. 光伏装机达到 9999 吉瓦"  # 9999 不在材料
        self.assertFalse(numeric_consistency(out, self.MAT))

    def test_full_material_fallback_for_short_excerpt(self):
        # 节选(input)限长截断恰好漏数字, 全文兜底放行
        excerpt = "非化石能源占比达到18.9%"
        full = "早在2020年, 目标已是18.9%。" + self.MAT
        # 节选内找不到 2025 → 不用全文时挂
        self.assertFalse(numeric_consistency("2025 年目标 18.9%", excerpt))
        # 用全文复查 → 放行
        self.assertTrue(numeric_consistency("2025 年目标 18.9%", excerpt,
                                            full_material=full))

    def test_no_numbers_passes(self):
        self.assertTrue(numeric_consistency("要点一。要点二。", self.MAT))

    def test_numbers_in_fullwidth(self):
        self.assertIn("18.9%", numbers_in("占比１８.９％"))

    def test_too_short_rejected(self):
        self.assertFalse(numeric_consistency("x", self.MAT))


if __name__ == "__main__":
    unittest.main()