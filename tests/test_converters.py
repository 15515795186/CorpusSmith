import json
import os
import unittest

from file_processor.converters.pretrain_converter import PretrainConverter
from file_processor.converters.instruction_converter import InstructionConverter
from file_processor.converters.cot_converter import COTConverter
from file_processor.readers.base import ReaderResult
from tests._tempdir import cleanup, workspace_tempdir

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


class COTJudgeTest(unittest.TestCase):
    """判类只在 SFT 样本层(《数据集介绍.md》4.5)——三态: 四类 / 四类之外 / 难以区分(人工审核)"""

    def setUp(self):
        self.cc = COTConverter()

    def _judge(self, instruction, output="材料给出的答案。"):
        return self.cc._judge_sample({"instruction": instruction, "output": output})

    def test_explicit_class_marker_wins(self):
        self.assertEqual(self._judge("请对比 ClickHouse 与 Loki 的查询能力。")[0], "比较")
        self.assertEqual(self._judge("计算 3 × 4 的结果。", "12")[0], "推理")
        self.assertEqual(self._judge("该机制的原理是什么？", "原理是热传导。")[0], "解释")

    def test_template_and_filename_stripped_before_judging(self):
        raw = "根据以下关于「比较_cnblogs_x.txt」的材料，请解释其原理。"
        cls, why = self._judge(raw)
        self.assertEqual(cls, "解释")
        self.assertEqual(why, "题面显式类别标记")
        stripped = self.cc._strip_template(raw)
        self.assertNotIn("cnblogs", stripped, "来源文件名不得进入判类依据")
        self.assertNotIn("比较", stripped)

    def test_counting_question_is_reasoning(self):
        self.assertEqual(self._judge("材料中共列出了几个条目？", "共 5 个。")[0], "推理")

    def test_numbered_list_without_counting_is_not_reasoning(self):
        doc = "一、加强统筹协调。\n二、完善政策体系。\n三、强化监督评估。"
        self.assertNotEqual(self._judge(doc)[0], "推理")

    def test_summary_and_fact_retrieval_are_out_of_scope(self):
        for q in ("请总结材料的主要内容。", "编制报告依据了哪些文件？",
                  "Identify the Python version mentioned in the text.",
                  "谁是本文的作者？", "When was this publication released?"):
            self.assertIsNone(self._judge(q)[0], q)

    def test_conflicting_explicit_markers_go_to_human_review(self):
        cls, why = self._judge("请对比 A 与 B 的差异，并给出改进建议。")
        self.assertEqual(cls, COTConverter.UNCERTAIN)
        self.assertIn("冲突", why)

    def test_empty_instruction_is_uncertain(self):
        self.assertEqual(self._judge("")[0], COTConverter.UNCERTAIN)


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




class COTJudgeBilingualTest(unittest.TestCase):
    """四类判类中英文同一套语义特征(英文形态缺失会让英文材料整块判空)"""

    def _judge(self, instruction, output="Answer from the material."):
        return COTConverter()._judge_sample({"instruction": instruction, "output": output})[0]

    def test_english_classes(self):
        self.assertEqual(self._judge("Compare the two methods described in the text."), "比较")
        self.assertEqual(self._judge("Why does convection form clouds?"), "解释")
        self.assertEqual(self._judge("What should forecasters do before issuing a warning?"), "建议")
        self.assertEqual(self._judge("Calculate the total cost: 12 + 30."), "推理")

    def test_english_retrieval_is_out_of_scope(self):
        for q in ("Who are the authors of this paper?", "List three categories below the display options.",
                  "What type of light does Webb detect?"):
            self.assertIsNone(self._judge(q), q)


class SkillTemplateReferenceTest(unittest.TestCase):
    """生成 prompt 必须嵌 skill 的**真实数据集模板原文**。

    回归点: COT 四类曾被我写成一句中文概述("真实样本风格(...)：..."),
    丢了真实样本的字段结构与格式标记 —— 规范给了样本 JSON, 就该原样用。
    """

    def test_cot_shots_keep_real_sample_structure(self):
        shots = COTConverter._COT_SHOT
        self.assertIn("<<48/2=24>>", shots["推理"][1])
        self.assertIn("#### 72", shots["推理"][1])
        self.assertIn("openai/gsm8k", shots["推理"][0])
        self.assertIn('"correct_answer"', shots["解释"][1])
        self.assertIn('"support"', shots["解释"][1])
        self.assertIn('"title"', shots["建议"][1])
        self.assertIn('"body"', shots["建议"][1])
        self.assertIn('"chosen"', shots["比较"][1])
        self.assertIn('"rejected"', shots["比较"][1])

    def test_cot_prompt_embeds_real_sample_both_languages(self):
        cc = COTConverter()
        for lang in ("zh", "en"):
            p = cc._build_cot_prompt("问题", "答案", "材料", lang, "建议")
            self.assertIn("legal-advice-reddit", p)
            self.assertIn('"title"', p)
            self.assertNotIn("真实样本风格(雇主裁员", p)   # 旧的中文转述不应再出现

    def test_sft_shots_reference_real_datasets(self):
        ic = InstructionConverter()
        p_zh = ic._build_instruction_prompt("文本", "zh")
        self.assertIn("alpaca-zh", p_zh)
        self.assertIn("三原色是什么？", p_zh)
        p_en = ic._build_instruction_prompt("text", "en")
        self.assertIn("tatsu-lab/alpaca", p_en)
        self.assertIn("Give three tips for staying healthy.", p_en)



class COTJudgeNarrowTest(unittest.TestCase):
    """判类窄化: 日期/数值区间不算算式; 编号列表必须有计数提问才算推理"""

    def setUp(self):
        self.cc = COTConverter()

    def _judge(self, instruction, output="材料给出的答案。"):
        return self.cc._judge_sample({"instruction": instruction, "output": output})[0]

    def test_date_and_range_are_not_math(self):
        for tx in ("发布日期 2024-03-19，同比增长 40-50%，应当加强统筹协调。",
                   "2022-2024 年累计投资 300-500 亿元，各地区需要健全考核评估机制。"):
            self.assertNotEqual(self._judge(tx), "推理", tx[:24])

    def test_real_math_is_reasoning(self):
        self.assertEqual(self._judge("某项目收益率 300/1200 = 0.25，请计算该比值。"), "推理")

    def test_list_needs_counting_question(self):
        no_cue = "\n".join(["一、加强统筹协调，明确责任分工。",
                            "二、完善政策体系，健全配套措施。",
                            "三、强化监督评估，确保落地见效。"])
        self.assertNotEqual(self._judge(no_cue), "推理")
        self.assertEqual(self._judge("请清点材料中一共列出了几项重点任务？", no_cue), "推理")

    def test_conditional_deduction_is_reasoning(self):
        self.assertEqual(
            self._judge("如果温度高于 25 摄氏度，那么设备就会启动保护程序。", "设备会启动保护程序。"),
            "推理")


class COTSampleDrivenTest(unittest.TestCase):
    """COT 以既有 SFT 问答对为唯一素材: 不自造问答对 + 字段契约 + 来源可溯"""

    class _FakeLLM:
        GOOD = ("1. 材料给出两个对象各自的强弱项，逐项核对后落在同一结论上。\n"
                "2. 综合以上事实收敛到回答。\n因此结论是: ClickHouse 更强。")

        def __init__(self, responses=None):
            self.responses = list(responses) if responses else [self.GOOD]
            self.calls = []

        def generate(self, prompt, system_prompt=None, **kw):
            self.calls.append(prompt)
            return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]

    SFT = {
        "id": "sft-000042",
        "instruction": "请对比 ClickHouse 与 Loki 的查询能力差异。",
        "input": "ClickHouse 查询能力强；Loki 查询较弱。",
        "output": "ClickHouse 更强。",
        "source": "doc.txt",
    }
    CORPUS = "前言段落。ClickHouse 查询能力强；Loki 查询较弱。后记段落。"

    def test_sample_is_basis_and_fields_filled(self):
        cc = COTConverter(llm_generator=self._FakeLLM())
        samples = cc.convert_sft_batch("doc.txt", [self.SFT], corpus_text=self.CORPUS)
        self.assertEqual(len(samples), 1)
        s = samples[0]
        self.assertEqual((s["instruction"], s["input"], s["output"]),
                         (self.SFT["instruction"], self.SFT["input"], self.SFT["output"]),
                         "素材不得自造: 三个字段必须沿用 SFT 原样")
        self.assertEqual(s["task_type"], "比较")
        self.assertTrue(s["chain_of_thought"].strip())
        self.assertFalse(s["verifiable"])
        self.assertIsNone(s["final_answer_correct"], "未验证不得填")
        self.assertEqual(s["source"]["sft_id"], "sft-000042")
        self.assertEqual(s["source"]["corpus_file"], "doc.txt")

    def test_context_retrieved_from_corpus(self):
        llm = self._FakeLLM()
        COTConverter(llm_generator=llm).convert_sft_batch("doc.txt", [self.SFT], corpus_text=self.CORPUS)
        self.assertIn("Loki 查询较弱", llm.calls[0], "回原文搜据的 context 必须进 prompt")

    def test_out_of_scope_sample_not_converted(self):
        rec = dict(self.SFT, instruction="请总结材料的主要内容。")
        cc = COTConverter(llm_generator=self._FakeLLM())
        self.assertEqual(cc.convert_sft_batch("doc.txt", [rec], corpus_text=self.CORPUS), [])
        self.assertEqual(cc.stats["out_of_scope"], 1)

    def test_uncertain_sample_exported_for_human_review(self):
        tmp = workspace_tempdir()
        try:
            review = os.path.join(tmp, "review.jsonl")
            rec = dict(self.SFT, instruction="请对比 A 与 B 的差异，并给出改进建议。")
            cc = COTConverter(llm_generator=self._FakeLLM(), review_output=review)
            self.assertEqual(cc.convert_sft_batch("doc.txt", [rec], corpus_text=self.CORPUS), [])
            rows = [json.loads(l) for l in open(review, encoding="utf-8") if l.strip()]
            self.assertEqual(len(rows), 1)
            self.assertIn("判类难以区分", rows[0]["_review_reason"])
        finally:
            cleanup(tmp)

    def test_reasoning_flags_only_when_machine_verified(self):
        rec = {"id": "sft-000007", "instruction": "计算 3 × 4 的结果。",
               "input": "材料：3 × 4 = 12", "output": "12", "source": "doc.txt"}
        llm = self._FakeLLM(["1. 依据材料给出的算式 3 × 4 = 12，逐步得到乘积 12。\n因此结论是: 12"])
        samples = COTConverter(llm_generator=llm).convert_sft_batch(
            "doc.txt", [rec], corpus_text="材料：3 × 4 = 12")
        self.assertEqual(len(samples), 1)
        s = samples[0]
        self.assertEqual(s["task_type"], "推理")
        self.assertTrue(s["verifiable"])
        self.assertTrue(s["final_answer_correct"], "算式重算通过才填")
        self.assertTrue(s["reasoning_correct"])

    def test_retry_then_drop(self):
        cc = COTConverter(llm_generator=self._FakeLLM(["短。", "短。", "短。"]), llm_max_attempts=3)
        self.assertEqual(cc.convert_sft_batch("doc.txt", [self.SFT], corpus_text=self.CORPUS), [])
        self.assertEqual(len(cc.llm_generator.calls), 3, "应重试到上限")
        self.assertEqual(cc.stats["dropped"], 1, "重试后仍不过 -> 丢弃")

    def test_distribution_reported_not_pruned(self):
        cc = COTConverter(llm_generator=self._FakeLLM())
        samples = cc.convert_sft_batch("doc.txt", [self.SFT], corpus_text=self.CORPUS)
        dist = cc.report_distribution(samples)
        self.assertEqual(dist, {"比较": 1}, "只统计: 不为凑比例改判或删样本")


if __name__ == "__main__":
    unittest.main()