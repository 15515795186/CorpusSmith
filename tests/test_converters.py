import re
import unittest

from file_processor.converters.pretrain_converter import PretrainConverter
from file_processor.converters.instruction_converter import InstructionConverter
from file_processor.converters.cot_converter import COTConverter
from file_processor.readers.base import ReaderResult

CHINESE_DOC = (
    "# 数据处理的流程\n\n"
    "数据处理通常包含多个步骤，每一步都需要严格校验。\n\n"
    "1. 采集原始数据并校验来源\n"
    "2. 清洗数据并统一格式\n"
    "3. 标注数据并划分训练集\n\n"
    "机器学习是人工智能的一个分支，它通过数据训练模型并优化参数。\n\n"
    "因为训练数据存在噪声，所以模型需要正则化来提升泛化能力。\n"
)


def make_result(text=CHINESE_DOC, filepath="doc.txt", file_type="txt"):
    return ReaderResult(text=text, filepath=filepath, file_type=file_type, success=True)


class PretrainConverterTest(unittest.TestCase):

    def test_chunked_output_has_meta(self):
        converter = PretrainConverter(chunk_size=100, chunk_overlap=10)
        samples = converter.convert(make_result())
        self.assertGreater(len(samples), 1)
        for sample in samples:
            self.assertIn("text", sample)
            self.assertEqual(sample["meta"]["source"], "doc.txt")
            self.assertEqual(sample["meta"]["type"], "txt")
            self.assertGreater(sample["meta"]["token_count_est"], 0)

    def test_no_chunk_mode_outputs_single_sample(self):
        converter = PretrainConverter(chunk_size=0)
        samples = converter.convert(make_result())
        self.assertEqual(len(samples), 1)
        self.assertIn("数据处理的流程", samples[0]["text"])

    def test_max_samples_applied(self):
        converter = PretrainConverter(chunk_size=80, max_samples=2)
        self.assertEqual(len(converter.convert(make_result())), 2)

    def test_failed_result_returns_empty(self):
        converter = PretrainConverter()
        failed = ReaderResult(text="", filepath="x.txt", file_type="txt",
                              success=False, error="boom")
        self.assertEqual(converter.convert(failed), [])


class InstructionConverterTest(unittest.TestCase):

    def test_template_samples_generated(self):
        converter = InstructionConverter()
        samples = converter.convert(make_result())
        self.assertTrue(samples, "应能从列表/定义结构生成指令样本")
        for sample in samples:
            self.assertEqual(sample["source"], "doc.txt")
            self.assertTrue(sample["instruction"].strip())
            self.assertTrue(sample["output"].strip())

    def test_max_samples_applied(self):
        converter = InstructionConverter(max_samples=2)
        self.assertLessEqual(len(converter.convert(make_result())), 2)


class COTConverterTest(unittest.TestCase):

    def test_step_and_causal_samples(self):
        converter = COTConverter()
        samples = converter.convert(make_result())
        self.assertTrue(samples, "应能从编号步骤/因果关系生成 COT 样本")
        for sample in samples:
            self.assertIn("chain_of_thought", sample)
            self.assertTrue(sample["chain_of_thought"].strip())
            self.assertTrue(sample["output"].strip())
            self.assertEqual(sample["source"], "doc.txt")

    def test_max_samples_applied(self):
        converter = COTConverter(max_samples=1)
        self.assertEqual(len(converter.convert(make_result())), 1)


class InstructionGroundingTest(unittest.TestCase):
    """input 字段语义契约：答案来自特定文档的样本必须开卷（材料在 input），
    否则是在教模型闭卷背诵/幻觉；自包含通识题允许 input 为空"""

    def test_heading_sample_is_grounded(self):
        samples = InstructionConverter().convert(make_result())
        reading = [s for s in samples if "材料" in s["instruction"]]
        self.assertTrue(reading, "标题段落应生成开卷阅读样本")
        for s in reading:
            self.assertTrue(s["input"].strip(), "阅读样本的 input 必须携带来源材料")
            self.assertIn("数据处理的流程", s["instruction"])

    def test_definition_sample_is_self_contained(self):
        samples = InstructionConverter().convert(make_result())
        defs = [s for s in samples if s["instruction"].startswith("什么是")]
        self.assertTrue(defs, "定义句应产出自包含通识样本")
        for s in defs:
            self.assertEqual(s["input"], "", "自包含通识样本 input 应为空")

    def test_key_points_numbered_and_capped(self):
        content = "\n\n".join(
            f"第{i}段的主旨句会被抽取为一条要点，后面还有大量补充论述内容。"
            for i in range(8))
        pts = InstructionConverter()._key_points(content)
        lines = pts.split("\n")
        self.assertLessEqual(len(lines), 5)
        self.assertTrue(all(l.startswith(f"{j + 1}. ") for j, l in enumerate(lines)),
                        "要点应编号且顺序正确")

    def test_definition_gate_truncates_and_strips(self):
        doc = ("（六）地震动参数区划图，是指以地震动参数为指标，"
               "将全国划分为不同抗震设防要求区域的图件。第十八条 本条例自公布之日起施行。\n\n"
               + "补充段落内容若干字，用于把文档撑到列表与标题路径都不触发的状态。" * 3)
        samples = InstructionConverter()._generate_from_templates(doc, make_result(doc))
        defs = [s for s in samples if s["instruction"].startswith("什么是")]
        self.assertEqual(len(defs), 1)
        self.assertEqual(defs[0]["instruction"], "什么是地震动参数区划图？")
        self.assertTrue(defs[0]["output"].endswith("图件。"))
        self.assertNotIn("第十八条", defs[0]["output"], "答案不得吞进下一条编号")
        self.assertEqual(defs[0]["input"], "")

    def test_definition_gate_rejects_verb_slices(self):
        doc = ("五、二十一世纪初的农村扶贫开发缓解和消除贫困仍然是指"
               "中国今后一项长期的历史任务。为加快解决贫困问题而召开中央扶贫会议。\n\n"
               "信息公开有两种方式：一是指每年新生入学时发放手册并开展宣讲活动。\n\n"
               + "其余段落与上述结构无关，仅用于维持文档整体长度与主题连续性说明。" * 3)
        samples = InstructionConverter()._generate_from_templates(doc, make_result(doc))
        defs = [s for s in samples if s["instruction"].startswith("什么是")]
        self.assertEqual(defs, [], "动词误接（仍然是指）与超长编号术语应被门控拦截")


class COTGroundingTest(unittest.TestCase):

    STEP_DOC = (
        "项目实施流程\n\n"
        "1. 需求调研并输出需求分析报告\n"
        "2. 方案设计并组织专家评审\n"
        "3. 编码实现并完成自测验证\n\n"
        "首先，项目组完成需求调研，其次，进入方案设计与评审，最后，完成编码实现与验收。\n"
    )

    def test_step_samples_are_grounded(self):
        samples = COTConverter().convert(make_result(self.STEP_DOC))
        steps = [s for s in samples if "根据以下材料" in s["instruction"]]
        self.assertTrue(steps, "步骤类样本应改为开卷形态")
        for s in steps:
            self.assertTrue(s["input"].strip())
            self.assertNotIn("该任务", s["instruction"], "无标题时不得硬造空指代")

    SECTION_DOC = (
        "论文结构\n\n"
        "1. Introduction\n\n"
        "2. Related Work\n\n"
        "3. Method\n\n"
        "4. Experiments\n\n"
        "5. Conclusion\n"
    )

    def test_section_numbering_is_not_a_process(self):
        samples = COTConverter().convert(make_result(self.SECTION_DOC))
        bogus = [s for s in samples if "步骤" in s["instruction"]]
        self.assertEqual(bogus, [], "论文章节编号不得被当作流程步骤生成样本")

    def test_causal_sample_input_stays_empty(self):
        samples = COTConverter().convert(make_result(CHINESE_DOC))
        causal = [s for s in samples if s["instruction"].startswith("根据已知信息")]
        self.assertTrue(causal)
        for s in causal:
            self.assertEqual(s["input"], "", "条件已在题面，自包含合规")

    def test_three_step_sample_grounded(self):
        samples = COTConverter().convert(make_result(self.STEP_DOC))
        three = [s for s in samples if "第一步骤" in s["chain_of_thought"]]
        self.assertTrue(three)
        for s in three:
            self.assertIn("首先", s["input"], "题面不得依赖看不见的原文")


class COTRecallExpansionTest(unittest.TestCase):
    """中文编号清点通道 + 因果关联词扩集：新增召回必须不破坏开卷契约"""

    WHITEPAPER = (
        "中国对非援助政策宣示\n\n"
        "一、扩大对非援助规模，到2009年使中国对非洲国家的援助规模比2006年增加一倍。\n"
        "二、三年内面向非洲国家提供三百亿美元的优惠贷款和二十亿美元的优惠出口买方信贷。\n"
        "三、为鼓励和支持中国企业到非洲投资，设立中非发展基金，基金总额逐步达到五十亿美元。\n"
    )

    def test_cn_numbered_block_yields_inventory_only(self):
        samples = COTConverter().convert(make_result(self.WHITEPAPER))
        inv = [s for s in samples if "清点" in s["instruction"]]
        self.assertEqual(len(inv), 1, "中文编号块应产出恰一条清点样本")
        self.assertIn("一、扩大对非援助规模", inv[0]["input"], "材料必须进 input（开卷）")
        self.assertIn("三、为鼓励和支持", inv[0]["input"])
        self.assertNotIn("完整流程", "".join(s["instruction"] for s in samples),
                         "无时序证据的中文编号（并行宣告）不得产流程样本")

    def test_cn_toc_short_titles_rejected(self):
        doc = "报告目录\n\n一、概述\n二、方法\n三、结果\n四、结论\n"
        samples = COTConverter().convert(make_result(doc))
        self.assertEqual([s for s in samples if "清点" in s["instruction"]], [],
                         "短标题目录不得生成清点样本（中位长≥12守卫）")

    def test_cn_count_matches_visible_material(self):
        # 12 条 × ~140 字 > 1500 预算：计数答案必须等于截断后材料里可见的条目数
        cn = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '十一', '十二']
        body = "\n".join(f"{c}、推进{p}领域的制度建设，形成配套措施并组织实施保障方案，确保落地见效。{'补充说明文字' * 25}"
                         for c, p in zip(cn, ['经济', '社会', '文化', '生态', '教育', '科技',
                                              '民生', '卫生', '住房', '就业', '安全', '法治']))
        samples = COTConverter().convert(make_result(body + "\n"))
        inv = [s for s in samples if "清点" in s["instruction"]]
        self.assertEqual(len(inv), 1)
        visible = sum(1 for ln in inv[0]["input"].split("\n")
                      if re.match(r'^[一二三四五六七八九十]{1,3}、', ln))
        claimed = int(re.search(r'列出(\d+)个条目', inv[0]["output"]).group(1))
        self.assertEqual(claimed, visible, "计数不得引用材料里看不到的条目")

    def test_paren_style_inventory(self):
        doc = ("（一）第一阶段的目标是基本完成主要制度的框架设计工作。\n"
               "（二）第二阶段重点推进试点地区的实施与评估反馈机制。\n"
               "（三）第三阶段总结推广成熟经验并建立长效运行保障体系。\n")
        samples = COTConverter().convert(make_result(doc))
        inv = [s for s in samples if "清点" in s["instruction"]]
        self.assertEqual(len(inv), 1)
        self.assertIn("三、第三阶段", inv[0]["input"], "括号编号归一为顿号样式入材料")
        self.assertIn("一、第一阶段", inv[0]["input"])

    def test_causal_narrative_and_swallow_rejected(self):
        doc = ("由于有人提出“我们看一看地图”的建议，因此会场短暂安静了下来。\n"
               "因为我们坚持了基本原则，所以会议取得了超出预期的阶段性成果；由于双方互谅互让，"
               "因此会谈最终签署了联合公报。\n")
        samples = COTConverter().convert(make_result(doc))
        causal = [s for s in samples if s["instruction"].startswith("根据已知信息")]
        self.assertEqual(causal, [], "引号嵌套（转述）与分号并环（多段推理链）不得切片成样本")

    def test_new_causal_connectives_accepted(self):
        doc = ("由于历史原因形成的两岸分离状态尚未消除，因此实现统一需要兼顾现实与法理两个维度。\n"
               "只有坚持一个中国原则的基础不动摇，才能为两岸协商创造必要的政治前提条件。\n")
        samples = COTConverter().convert(make_result(doc))
        causal = [s for s in samples if s["instruction"].startswith("根据已知信息")]
        self.assertEqual(len(causal), 2, "由于…因此 / 只有…才能 都应被识别")
        for s in causal:
            self.assertEqual(s["input"], "", "题面自包含（条件已写入），留空合规")
        self.assertTrue(any("才能" in s["output"] or s["output"].startswith("为两岸")
                            for s in causal), "只有…才能 的结论不得残留'能'字错切")

    def test_causal_anaphora_and_cross_sentence_rejected(self):
        doc = ("因为这家长期坚持高研发投入的科技型企业，所以获得了资本市场的持续关注。\n"
               "因为外部市场需求趋于饱和，库存持续积压。产业政策收紧，所以企业必须加快转型。\n")
        samples = COTConverter().convert(make_result(doc))
        causal = [s for s in samples if s["instruction"].startswith("根据已知信息")]
        self.assertEqual(causal, [],
                         "指代词开头的前提（悬空）与含句号的跨句贪婪配对都必须拒绝")


class NumberedListSampleTest(unittest.TestCase):
    """白皮书式编号条目（并行措施）的正确落点：instruction 列举任务，而非 COT 流程"""

    WHITEPAPER = (
        "中国对非援助政策宣示\n\n"
        "一、扩大对非援助规模，到2009年使中国对非洲国家的援助规模比2006年增加一倍。\n"
        "二、三年内面向非洲国家提供三百亿美元的优惠贷款和二十亿美元的优惠出口买方信贷。\n"
        "三、为鼓励和支持中国企业到非洲投资，设立中非发展基金，基金总额逐步达到五十亿美元。\n"
    )

    def test_numbered_block_becomes_enumeration_sample(self):
        doc = self.WHITEPAPER
        samples = InstructionConverter()._generate_from_templates(doc, make_result(doc))
        enum = [s for s in samples if "逐条列举并编号" in s["instruction"]]
        self.assertTrue(enum)
        self.assertIn("中非发展基金", enum[0]["input"], "材料必须进 input（开卷）")
        self.assertEqual(enum[0]["output"].count("\n"), 2, "输出应为 3 条编号")

    def test_toc_like_block_not_enumerated(self):
        doc = "报告目录\n\n一、概述\n二、方法\n三、结果\n四、结论\n"
        samples = InstructionConverter()._generate_from_templates(doc, make_result(doc))
        enum = [s for s in samples if "逐条列举并编号" in s["instruction"]]
        self.assertEqual(enum, [], "短标题目录不应生成列举样本")


if __name__ == "__main__":
    unittest.main()