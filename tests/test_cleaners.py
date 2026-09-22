import unittest

from file_processor.cleaners.text_cleaner import TextCleaner
from file_processor.cleaners.quality_filter import QualityFilter
from file_processor.utils.dedup import Deduplicator, SimHash
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


class SimHashTest(unittest.TestCase):
    """去重指纹回归测试

    历史缺陷1：_hash 用 Python 内置 hash()，其随机盐（PYTHONHASHSEED）导致
    同一语料在不同进程判重结果不同（实测 12 篇 vs 8 篇，交集仅 4 篇）。
    历史缺陷2：token 用单字/单词，同领域文档词表高度重叠，
    真实语料误杀 18/50（白皮书）与 12/50（arXiv 论文）。改用 bigram
    （中文相邻字 + 英文相邻词）后，5 类语料实测仅真实再版/模板件被识别。
    历史缺陷3：只看指纹相似度会把同模板不同填充的文档（不同高校简介页，
    长度差异大）误判重复，故增加长度比约束。
    """

    GOLDEN_FP = 12472177444965147942  # CHINESE_TEXT 的稳定指纹（blake2b+zh-bigram）

    IRRIGATION = (
        "本文系统研究了干旱区灌溉制度的演变过程，并通过多年试验数据对比了节水灌溉"
        "在不同土壤类型上的产量响应，结果表明滴灌处理显著提高了水分利用效率。")
    ANNUAL_REPORT = (
        "公司年度报告显示本年度营业收入增长百分之十二，净利润达到三点五亿元，"
        "海外业务扩张速度加快，管理层计划在未来三年追加研发投入占比。")

    def test_fingerprint_is_stable_across_processes(self):
        """内置 hash() 会让此值随 PYTHONHASHSEED 变化 → 该断言即防回归"""
        self.assertEqual(SimHash().fingerprint(CHINESE_TEXT), self.GOLDEN_FP)

    def test_unrelated_docs_not_duplicate(self):
        dedup = Deduplicator()
        self.assertFalse(dedup.is_duplicate(self.IRRIGATION * 3))
        self.assertFalse(dedup.is_duplicate(self.ANNUAL_REPORT * 3),
                         "同领域不相关文档不应被误判为重复")

    def test_small_edit_detected_as_duplicate(self):
        dedup = Deduplicator()
        self.assertFalse(dedup.is_duplicate(self.IRRIGATION * 3))
        edited = self.IRRIGATION.replace("产量响应", "亩产响应", 1) * 3
        self.assertTrue(dedup.is_duplicate(edited), "轻微改动的再版文档应被识别")

    def test_length_ratio_guard(self):
        """同开头但长度差 > 阈值的文档不判重复（同模板不同填充）"""
        dedup = Deduplicator()
        short_doc = self.IRRIGATION * 3
        long_doc = self.IRRIGATION * 12  # 长度比 0.25 < min_length_ratio
        self.assertFalse(dedup.is_duplicate(short_doc))
        self.assertFalse(dedup.is_duplicate(long_doc))

    def test_token_floor(self):
        """几乎无中英文内容的文本（如纯数字表）指纹无意义，不参与判重"""
        dedup = Deduplicator()
        digits_a = "1234 5678 9012 " * 10
        digits_b = "1234 5678 9012 " * 10
        self.assertFalse(dedup.is_duplicate(digits_a))
        self.assertFalse(dedup.is_duplicate(digits_b))

    def test_short_text_below_floor_not_deduped(self):
        dedup = Deduplicator()
        self.assertFalse(dedup.is_duplicate("短文本"))


class WikiUIBoilerplateTest(unittest.TestCase):
    """质检实测反馈：维基文库下载件带导航/页脚/[编辑] 残留，
    被评分模型点名为语言规范性缺陷 → 属于清洗器遗漏而非数据问题"""

    def setUp(self):
        self.cleaner = TextCleaner()

    def test_nav_word_lines_removed(self):
        text = ("跳转到内容 导航 搜索\n"
                "阅读 编辑 查看历史\n"
                "第一章 总则\n"
                "本条例自公布之日起施行，研究人员对其进行了搜索与比对验证。")
        cleaned = self.cleaner.clean(text)
        self.assertNotIn("跳转到内容", cleaned)
        self.assertNotIn("查看历史", cleaned)
        self.assertIn("第一章 总则", cleaned)
        # 含"搜索"的正常内容行不得误删
        self.assertIn("进行了搜索与比对验证", cleaned)

    def test_footer_and_edit_markers_removed(self):
        text = ("《中华人民共和国妇女权益保障法》[编辑]\n"
                "本作品翻译自公有领域文件。\n"
                "第一条 为了保障妇女的合法权益，制定本法。")
        cleaned = self.cleaner.clean(text)
        self.assertNotIn("[编辑]", cleaned)
        self.assertIn("第一条", cleaned)

    def test_centralnotice_removed(self):
        text = "CentralNotice zh-CN\n正文内容从这里开始，包含完整的句子。是的。"
        cleaned = self.cleaner.clean(text)
        self.assertNotIn("CentralNotice", cleaned)
        self.assertIn("正文内容", cleaned)

    def test_wiki_category_tag_removed(self):
        """质检点名：'…亚太安全Category:釣魚臺列嶼' 紧贴中文的分类标签也须清除"""
        text = ("中国将推动亚太安全Category:釣魚臺列嶼\n"
                "Category: 政治安全\n"
                "对话合作机制不断完善。")
        cleaned = self.cleaner.clean(text)
        self.assertNotIn("Category", cleaned)
        self.assertIn("亚太安全", cleaned)
        self.assertIn("对话合作机制不断完善", cleaned)

    def test_skin_phrases_removed_from_merged_line(self):
        """格式框/PD 声明被 reader 并进正文段落时，须按短语级清除"""
        text = ("第八十六条\n\n"
                "本法自2023年1月1日起施行。本作品来自中华人民共和国法律。"
                "依据《中华人民共和国著作权法》第五条，本作品不适用于该法，"
                "所以属于公有领域。 EPUB 用于电脑、手机、平板 MOBI 用于Kindle\n\n"
                "下载中华人民共和国妇女权益保障法")
        cleaned = self.cleaner.clean(text)
        self.assertNotIn("公有领域", cleaned)
        self.assertNotIn("EPUB", cleaned)
        self.assertNotIn("Kindle", cleaned)
        self.assertIn("本法自2023年1月1日起施行", cleaned)
        self.assertIn("中华人民共和国妇女权益保障法", cleaned)


if __name__ == "__main__":
    unittest.main()