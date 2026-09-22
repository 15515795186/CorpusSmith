import unittest

from file_processor.cleaners.text_cleaner import TextCleaner
from file_processor.cleaners.quality_filter import QualityFilter
from file_processor.utils.lang_detect import detect_language

CHINESE_TEXT = (
    "机器学习是人工智能的一个重要分支，它通过数据训练模型并不断优化参数，"
    "从而在自然语言处理、计算机视觉与语音识别等领域取得显著进展。"
    "深度学习依赖多层神经网络自动提取特征，减少了人工设计特征的繁琐过程。"
    "强化学习则通过与环境交互获得奖励信号来优化策略，在机器人控制场景中被广泛采用。"
)


class TextCleanerTest(unittest.TestCase):

    def setUp(self):
        self.cleaner = TextCleaner()

    def test_chinese_punctuation_is_preserved(self):
        cleaned = self.cleaner.clean(CHINESE_TEXT)
        self.assertIn("。", cleaned, "中文句号不应被替换成英文句点")
        self.assertIn("，", cleaned, "中文逗号不应被替换成英文逗号")
        self.assertNotIn("机器学习是人工智能的一个重要分支,", cleaned)

    def test_english_punctuation_is_normalized(self):
        cleaned = self.cleaner.clean("Machine learning is useful。Really。", language="en")
        self.assertNotIn("。", cleaned)
        self.assertIn(".", cleaned)

    def test_control_chars_removed(self):
        cleaned = self.cleaner.clean("abc\x00\x07def", language="en")
        self.assertNotIn("\x00", cleaned)
        self.assertIn("abcdef", cleaned.replace(" ", ""))

    def test_urls_replaced(self):
        cleaned = self.cleaner.clean("see https://example.com/page for details", language="en")
        self.assertIn("[URL]", cleaned)
        self.assertNotIn("example.com", cleaned)

    def test_repeated_chars_collapsed(self):
        cleaned = self.cleaner.clean("wow!!!!!!!!!!", language="en")
        self.assertNotIn("!!!!!", cleaned)

    def test_empty_input(self):
        self.assertEqual(self.cleaner.clean(""), "")

    def test_numbered_steps_keep_line_breaks(self):
        text = "1. 采集原始数据并校验来源\n2. 清洗数据并统一格式\n3. 标注数据并划分训练集"
        cleaned = self.cleaner.clean(text)
        self.assertIn("\n2.", cleaned, "编号步骤不应被合并成一行")
        self.assertIn("\n3.", cleaned)

    def test_bullet_list_keeps_line_breaks(self):
        text = "- 数组与链表\n- 栈与队列\n- 哈希表与树"
        cleaned = self.cleaner.clean(text)
        self.assertIn("\n- 栈与队列", cleaned)

    def test_wrapped_lines_are_still_merged(self):
        text = "这是一段被错误断行的中文文本，它本\n来应该在同一行内显示完整。"
        cleaned = self.cleaner.clean(text)
        self.assertIn("它本来应该在", cleaned)


class QualityFilterTest(unittest.TestCase):

    def setUp(self):
        self.filter = QualityFilter(min_length=30, max_length=32768,
                                    garbage_ratio=0.3, dedup_threshold=0.85)

    def test_good_text_passes(self):
        result = self.filter.check(CHINESE_TEXT)
        self.assertTrue(result.passed, result.reason)

    def test_short_text_rejected(self):
        result = self.filter.check("太短")
        self.assertFalse(result.passed)
        self.assertIn("too_short", result.reason)

    def test_too_long_text_rejected(self):
        short_filter = QualityFilter(min_length=30, max_length=20)
        result = short_filter.check("字" * 40)
        self.assertFalse(result.passed)
        self.assertIn("too_long", result.reason)

    def test_garbage_text_rejected(self):
        garbage = "".join(chr(0xE000 + i % 200) for i in range(200))
        result = self.filter.check(garbage)
        self.assertFalse(result.passed)

    def test_repeated_char_run_rejected(self):
        text = "正常的开头内容。" + "啊" * 60 + "结尾的普通文本内容，用于保证长度足够。"
        result = self.filter.check(text)
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "garbage_content")

    def test_common_chinese_char_is_not_garbage(self):
        """高频汉字（的/国…）在 2000 字里出现 100+ 次是正常的，不能误判为乱码"""
        sentence = "我国的人民代表大会制度是中国的根本政治制度，它保障了人民的权利。"
        text = sentence * 40
        text = text[:2000] + "补充结尾内容，确保整体长度超过两千字并且语义正常。" * 30
        result = self.filter.check(text)
        self.assertTrue(result.passed, f"正常中文被误判: {result.reason}")

    def test_long_document_passes_with_default_max_length(self):
        """长文档（60KB+）不应因 max_text_length 被整篇丢弃，交由分块处理"""
        default_filter = QualityFilter()
        long_text = ("机器学习与深度学习在自然语言处理领域取得了显著进展。"
                     "本文提出了一种新的方法用于提升模型的泛化能力。") * 1200
        self.assertGreater(len(long_text), 32768)
        self.assertTrue(default_filter.check(long_text).passed)

    def test_explicit_max_length_still_enforced(self):
        strict = QualityFilter(min_length=30, max_length=1000)
        self.assertIn("too_long", strict.check("字" * 2000).reason)

    def test_duplicate_text_rejected(self):
        self.assertTrue(self.filter.check(CHINESE_TEXT).passed)
        second = self.filter.check(CHINESE_TEXT)
        self.assertFalse(second.passed)
        self.assertEqual(second.reason, "duplicate_content")

    def test_reset_clears_dedup_state(self):
        self.filter.check(CHINESE_TEXT)
        self.filter.reset()
        self.assertTrue(self.filter.check(CHINESE_TEXT).passed)


class LangDetectTest(unittest.TestCase):

    def test_chinese_detected(self):
        self.assertEqual(detect_language(CHINESE_TEXT), "zh")

    def test_english_detected(self):
        self.assertEqual(detect_language("Machine learning is a branch of AI."), "en")

    def test_empty_is_unknown(self):
        self.assertEqual(detect_language("   "), "unknown")


if __name__ == "__main__":
    unittest.main()