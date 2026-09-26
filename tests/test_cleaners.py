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


class FormTemplateFilterTest(unittest.TestCase):
    """质检淘汰复盘发现：空申报/审批表模板混在候选集里白耗质检调用。
    阈值在 400 文件真实语料标定：248 个通过文档零误伤，命中 4 个空表单。"""

    def setUp(self):
        self.filter = QualityFilter(min_length=30, max_length=1000000)

    def test_blank_form_grid_rejected(self):
        row = "姓名 | | 性别 | | 民族 | | 出生年月 | | 政治面貌 | |"
        text = "\n\n".join([row] * 30 + ["申报单位（盖章）：", "年 月 日"])
        self.assertGreater(len(text), 1000)
        result = self.filter.check(text)
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "form_template_grid")

    def test_dup_echo_grid_rejected(self):
        # 合并单元格复读型：几乎无空格子，靠 dup 分支拦截
        row = "身份证号 | 身份证号 | 身份证号 | 33020519**4517 | 33020519**4517"
        text = "\n".join([row] * 30)
        self.assertGreater(len(text), 1000)
        result = self.filter.check(text)
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "form_template_grid")

    def test_filled_roster_passes(self):
        # 真实名单：格子填满、含长实体名（叙述占比高），必须放行
        rows = [f"{i} | 张三丰{i} | 东莞市莞城运河小学分校{i} | "
                f"河北省石家庄市外国语实验中学教研处 | GP2307064000{i}"
                for i in range(1, 26)]
        text = "优秀学员名单\n\n" + "\n".join(rows)
        self.assertGreater(len(text), 1000)
        self.assertTrue(self.filter.check(text).passed)

    def test_small_grid_under_length_guard_passes(self):
        # 短网格（<1000 字符）交给 LLM 判断，启发式不拦截
        row = "学校 | | 项目 | |"
        text = "\n".join([row] * 25)
        self.assertLess(len(text), 1000)
        result = self.filter.check(text)
        self.assertNotEqual(result.reason, "form_template_grid")


class ExtractionNoiseTest(unittest.TestCase):
    """提取噪声修复回归（质检淘汰复盘驱动）

    本轮全量跑出 21 份淘汰样本，逐条核查理由后确认大部分是**提取/清洗缺陷**而非数据缺陷：
      · 8/10 份英文 arXiv 文字层 PDF 被判"OCR 错误(taki ing / sett tings / J¨urgen)"
        → 根因是清洗器断行规则把 "tak-\ning" 拼成 "taki" + 残行 "ng"（全库 4299 处跨行拼接候选）
      · 4 份政务网页稿 txt 被判"混入大量 HTML/网页渲染残留(&nbsp;&gt;&nbsp;)"
      · 2 份截图 OCR 被判"大量HTML标签和data-bbox属性残留"
    另有 22/80 份带连字(ﬁ/ﬂ)与坏变音符残留。以下用例逐条守住修复。
    """

    def setUp(self):
        self.cleaner = TextCleaner()

    # ---- 断词（历史 bug: r'(\w)-\n(\w)' → r'\1\2\n\2' 只复制换行后一个字母）----
    def test_hyphenated_line_break_merged(self):
        out = self.cleaner.clean("formed by tak-\ning a random matrix", language="en")
        self.assertIn("taking a random", out)

    def test_no_fake_word_fragment_left(self):
        """旧实现产出 "taki ing"（质检当成 OCR 错误点名）"""
        out = self.cleaner.clean("formed by tak-\ning a random matrix", language="en")
        self.assertNotIn("taki ing", out)
        self.assertNotIn("\nng", out)

    def test_real_hyphen_compound_kept(self):
        self.assertIn("COVID-19",
                      self.cleaner.clean("the COVID-\n19 pandemic", language="en"))
        self.assertIn("self-attention",
                      self.cleaner.clean("the self-\nattention layer", language="en"))

    def test_prefix_whitelist_is_narrow(self):
        """白名单只留 self-/non-/multi-/cross-: 虚词(in/be/an/re)是长词起始片段,
        全库审计 115 处"保留"里约 95 处是假保留(in-formation / be-tween / an-swer)"""
        self.assertIn("information",
                      self.cleaner.clean("the in-\nformation is", language="en"))
        self.assertIn("language",
                      self.cleaner.clean("the lan-\nguage model", language="en"))
        self.assertIn("non-parametric",
                      self.cleaner.clean("a non-\nparametric test", language="en"))

    # ---- 连字与坏变音符 ----
    def test_ligatures_expanded(self):
        out = self.cleaner.clean("di\ufb00erent and e\ufb03cient", language="en")
        self.assertIn("different", out)
        self.assertIn("efficient", out)

    def test_broken_diacritic_repaired(self):
        out = self.cleaner.clean("J\u00a8urgen Kurths", language="en")
        self.assertIn("Jürgen", out)

    def test_math_accents_untouched(self):
        """ˆξ / ˜a 是 LaTeX 数学帽子与波浪号, 不能映射成字母"""
        out = self.cleaner.clean("denote \u02c6\u03be and \u02dca as estimates", language="en")
        self.assertIn("\u02c6\u03be", out)
        self.assertIn("\u02dca", out)

    def test_private_use_glyphs_removed(self):
        """截图 OCR 把图标字体渲染成 U+E6xx 字形"""
        out = self.cleaner.clean("收藏\ue68a 打印\ue603 正文内容在这里。", language="zh")
        self.assertNotIn("\ue68a", out)
        self.assertIn("正文内容在这里", out)

    # ---- HTML 实体 / 网页稿 / OCR 转储 ----
    def test_html_entities_decoded(self):
        # 实体必须解码: 非面包屑位置的 ">" 正常出现, 实体本身不得残留
        out = self.cleaner.clean("见&nbsp;&gt;&nbsp;第 3 节", language="zh")
        self.assertIn(">", out)
        self.assertNotIn("&nbsp;", out)
        self.assertNotIn("&gt;", out)
        self.assertIn("第 3 节", out)

    def test_breadcrumb_line_dropped(self):
        '''政务门户面包屑整行删除; 正文里的 '>' 不受影响'''
        out = self.cleaner.clean("首页&nbsp;&gt;&nbsp;政策文件", language="zh")
        self.assertEqual(out.strip(), "")
        kept = self.cleaner.clean("见 > 第 3 节的内容说明。", language="zh")
        self.assertIn("见 > 第 3 节", kept)

    def test_snapshot_metadata_header_stripped(self):
        text = ("来源: https://zwgk.example.gov.cn/x.html\n"
                "标题: 202405_某通知\n\n"
                "国务院办公厅关于印发《某方案》的通知")
        out = self.cleaner.clean(text)
        self.assertNotIn("来源:", out)
        self.assertNotIn("202405_某通知", out)
        self.assertIn("国务院办公厅关于印发", out)

    def test_site_chrome_stripped(self):
        out = self.cleaner.clean("关于印发《某方案》的通知_国务院部门文件_中国政府网EN个人中心退出邮箱无障碍EN")
        self.assertNotIn("中国政府网", out)
        self.assertNotIn("个人中心", out)
        self.assertIn("关于印发《某方案》的通知", out)

    def test_ocr_markup_dump_unwrapped(self):
        text = ('```html\n<html><body>\n'
                '<div class="line" data-bbox="58 31 307 52">中华人民共和国中央人民政府</div>\n'
                '<div class="line" data-bbox="1 2 3 4">www.gov.cn</div>\n'
                '</body></html>')
        out = self.cleaner.clean(text)
        self.assertNotIn("data-bbox", out)
        self.assertNotIn("<div", out)
        self.assertIn("中华人民共和国中央人民政府", out)

    def test_legit_angle_brackets_preserved(self):
        """反例保护: 论文正文里的 <API>/<ATP> 是合法内容, 不得当作 HTML 残留清除"""
        text = "e(c) = <API> ac(ic) </API> and a Lean proof ends with <ATP>"
        out = self.cleaner.clean(text, language="en")
        self.assertIn("<API>", out)
        self.assertIn("<ATP>", out)

    def test_plain_chinese_text_untouched_by_new_rules(self):
        """新规则不得污染正常中文语料"""
        text = "机器学习是人工智能的重要分支，它通过数据训练模型并优化参数。"
        self.assertEqual(self.cleaner.clean(text), text)




class PortalResidueTest(unittest.TestCase):
    """政务门户页面残留(实测取证: 7~8 份页脚块、14~18 份元数据网格; 均整块重复两遍)"""

    FOOTER = ("关于本网网站声明联系我们网站纠错主办单位：国务院办公厅 运行维护单位： 运行中心"
              "版权所有： 中文域名： .政务网站标识码bm01001 京ICP备05070218号 "
              "京公网安备11010202001号国务院客户端国务院客户端小程序微博、微信电脑版客户端小程序"
              "微博微信注册网站纠错主办单位：国务院办公厅 运行维护单位： 运行中心版权所有： "
              "中文域名： .政务网站标识码bm01001京ICP备05070218号 京公网安备11010202001号回到顶部注册")

    def setUp(self):
        self.c = TextCleaner()

    def test_portal_footer_line_dropped(self):
        out = self.c.clean("政务要闻正文示例，说明有关工作的总体安排。\n\n" + self.FOOTER + "\n\n后续说明文字。")
        self.assertNotIn("网站标识码", out)
        self.assertNotIn("京ICP备", out)
        self.assertNotIn("国务院客户端", out)
        self.assertIn("政务要闻正文示例", out)

    def test_footer_link_wall_and_top_dropped(self):
        line = "链接：全国人大全国政协国家监察委员会最高人民法院最高人民检察院国务院部门网站地方政府网站驻港澳机构网站驻外机构"
        out = self.c.clean("正文内容示例。\n" + line + "\n结尾。")
        self.assertNotIn("驻外机构", out)
        self.assertIn("正文内容示例", out)

    def test_split_footer_fragment_lines_dropped(self):
        """整块页脚在 txt/html 里常被拆成多行, 每行只带一个标记"""
        doc = ("正文段落示例文字。\n驻外机构\n关于本网\n网站声明\n"
               "主办单位：国务院办公厅 运行维护单位：中国政府网运行中心\n"
               "网站标识码bm01000001 京ICP备05070218号 京公网安备11010202000001号\n"
               "国务院客户端\n网站标识码bm01001\n回到顶部\n后续正文段落。")
        out = self.c.clean(doc)
        for frag in ("驻外机构", "关于本网", "网站声明", "网站标识码", "京ICP备", "国务院客户端", "回到顶部"):
            self.assertNotIn(frag, out, frag)
        self.assertIn("正文段落示例文字", out)
        self.assertIn("后续正文段落", out)

    def test_nav_link_lines_dropped(self):
        """footer 侧栏链接行(整行只由机构名构成, 常带 - | ### 前缀)"""
        doc = ("正文示例。\n- | 最高人民检察院国务院部门网站\n- | 地方政府网站\n"
               "- | 驻港澳机构网站\n### 医务部门网站 | 地方政府网站 | 驻港澳机构网站 | 驻外机构\n"
               "地方政府网站管理办法由本级政府办公厅负责解释。")
        out = self.c.clean(doc)
        self.assertNotIn("驻港澳机构网站", out)
        self.assertNotIn("驻外机构", out)
        self.assertIn("地方政府网站管理办法", out, "含实义的正文行不得被删")

    def test_weak_footer_word_alone_kept(self):
        """只出现一个弱标记(联系我们)的正常句子不得被删"""
        line = "如需了解申报细节，请联系我们并查阅办事指南第三部分。"
        out = self.c.clean(line)
        self.assertIn("办事指南", out)

    def test_inline_metadata_grid_stripped(self):
        grid = ("标 题：工业和信息化部等九部门关于印发《精细化工产业创新发展实施方案（2024—2027年）》的通知"
                "发文机关：工业和信息化部 国家发展改革委 财政部 生态环境部"
                "发文字号：工信部联原〔2024〕136号来 源：工业和信息化部网站"
                "主题分类：工业、交通\\其他公文种类：通知成文日期：2024年07月02日")
        out = self.c.clean(grid + "\n\n各省、自治区、直辖市主管部门：现将方案印发给你们。")
        for lab in ("标 题：", "发文机关：", "发文字号：", "来 源：", "主题分类：", "成文日期："):
            self.assertNotIn(lab, out)
        self.assertIn("现将方案印发给你们", out)

    def test_legit_source_line_kept(self):
        """正文里的"数据来源：…"是单标签, 不得被当成页面网格删掉"""
        s = "本年度统计口径调整如下。数据来源：国家统计局年度公报，2024。"
        out = self.c.clean(s)
        self.assertIn("国家统计局", out)

    def test_repeated_label_line_keeps_first_only(self):
        doc = ("### 索引号：000014349/2024-00028\n### 发文机关：国务院办公厅\n"
               "### 索引号：000014349/2024-00028\n### 发文机关：国务院办公厅\n"
               "正文：国务院办公厅关于印发有关行动方案的通知。")
        out = self.c.clean(doc)
        # 网页字段表(标签行/值行)整体删除; 正文必须留存
        self.assertEqual(out.count("索引号"), 0, "字段表标签行应整行删")
        self.assertEqual(out.count("发文机关"), 0)
        self.assertIn("正文", out)

    def test_bare_ui_label_lines_dropped(self):
        """只剩界面标签、不带值的行(截图 OCR / markdown 化后常见)"""
        doc = "正文段落示例文字，说明有关工作要求。\n链接：\n**下载链接：**\n#### 发文字号\n后续正文段落。"
        out = self.c.clean(doc)
        self.assertNotIn("下载链接", out)
        self.assertNotIn("发文字号", out)
        self.assertIn("正文段落示例文字", out)
        self.assertIn("后续正文段落", out)

    def test_bare_size_word_must_not_corrupt_body(self):
        """裸词"超大"曾把正文改成"我国规模市场优势" —— 断言正文完好"""
        s = "要充分发挥我国超大规模市场优势，巩固外资在华发展信心。"
        out = self.c.clean(s)
        self.assertIn("超大规模市场优势", out)

    def test_zhanghao_variants_removed(self):
        for v in ("字号：默认超大", "字号：默认大更大", "字号：默认大", "字号：默认更大"):
            out = self.c.clean(f"工业和信息化部关于印发有关方案的通知{v}\n\n正文内容。")
            self.assertNotIn("默认", out, v)

    def test_long_duplicate_line_deduped_short_kept(self):
        longline = "各省、自治区、直辖市及计划单列市工业和信息化主管部门，有关中央企业：现将实施方案印发给你们，请结合实际认真抓好落实。"
        # 首行须以句末标点收尾, 否则会被断行修复与下一行合并成不同文本, 掩盖重复
        doc = "标题行。\n" + longline + "\n" + longline + "\n" + "①②\n" + "①②\n"
        out = self.c.clean(doc)
        self.assertEqual(out.count("现将实施方案印发给你们"), 1, "完全相同的长行只保留首次")
        self.assertEqual(out.count("①②"), 2, "短行(未达 30 字符)不做去重")


if __name__ == "__main__":
    unittest.main()