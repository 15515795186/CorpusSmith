# -*- coding: utf-8 -*-
"""列表序号 vs 可核数值: 序号不得被当成"链中数值", 真数值不得被吃掉"""
import re
import unittest

from file_processor.converters.cot_converter import COTConverter
from file_processor.utils.validation import numbers_in


class ListMarkerNumberGuardTest(unittest.TestCase):

    def test_line_start_marker_stripped(self):
        chain = "1. 先核对材料。\n2. 再收敛结论。"
        bare = COTConverter._strip_list_markers(chain)
        self.assertIsNone(re.search(r"\d", bare), bare)

    def test_inline_marker_after_colon_stripped(self):
        chain = ("材料说明如下。\nthe answer is: 1. Loki 的查询功能较弱，不支持复杂查询。\n"
                 "2. Elasticsearch 支持复杂的全文搜索。")
        bare = COTConverter._strip_list_markers(chain)
        self.assertIsNone(re.search(r"\d", bare), bare)
        self.assertEqual(numbers_in(bare), set())

    def test_real_numbers_and_decimals_kept(self):
        chain = "计算得到 3.5 与 42 两个结果，另有 12）项说明。"
        bare = COTConverter._strip_list_markers(chain)
        nums = numbers_in(bare)
        self.assertIn("3.5", str(nums), nums)
        self.assertIn("42", str(nums), nums)

    def test_chinese_marker_stripped(self):
        chain = "一、先看材料。\n二、再看问答对。"
        bare = COTConverter._strip_list_markers(chain)
        self.assertIsNone(re.search(r"\d", bare), bare)


if __name__ == "__main__":
    unittest.main()
