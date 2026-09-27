"""拒绝采样(失败重试制)单测
硬校验不过才重试(llm_max_attempts); instruction 按 chunk 重试,
COT 只对推理类重试(算式重算/数字一致性可机判), 建议/比较/解释维持单次。
全部用 FakeGenerator, 不触发真实 LLM。
"""
import unittest
from types import SimpleNamespace

from file_processor.converters.instruction_converter import InstructionConverter
from file_processor.converters.cot_converter import COTConverter


class FakeGenerator:
    """按序返回预置响应; 用尽后复用最后一个。"""

    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
        self.prompts = []          # 记录每次 prompt(按类统计调用次数用)

    def generate(self, prompt, system_prompt=None):
        r = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        self.prompts.append(prompt)
        return r


def make_result(text: str) -> SimpleNamespace:
    return SimpleNamespace(success=True, text=text, filepath="test.txt")


# 5 个编号条目(内容≥10字符, 符合 2b 列举块条件): 纯规则确定性, 供 COT 通道复用
LIST_TEXT = (
    "(一)加强煤炭清洁高效利用，推动煤电低碳化改造。\n"
    "(二)严格实施大气污染防治重点区域煤炭消费总量控制。\n"
    "(三)重点削减非电力用煤，持续提升电力用煤质量。\n"
    "(四)到2025年底，大气污染防治重点区域平原地区散煤基本清零。\n"
    "(五)推进煤电节能降碳改造、灵活性改造、供热改造三改联动。\n"
)


class TestInstructionRetry(unittest.TestCase):
    def test_retries_when_first_round_fully_invalid(self):
        # 首次返回空数组(解析 0 条) → 重试; 第二次正常 → 停止
        good = ('[{"instruction": "材料中的第五项措施是什么?",'
                ' "input": "",'
                ' "output": "第五项措施与能源政策相关。"}]')
        gen = FakeGenerator(["[]", good])
        ic = InstructionConverter(llm_generator=gen, llm_max_attempts=3)
        samples = ic._generate_with_llm(LIST_TEXT, make_result(LIST_TEXT))
        self.assertEqual(gen.call_count, 2)      # 重试发生
        self.assertGreaterEqual(len(samples), 1)  # 最终拿到有效样本

    def test_no_retry_when_first_round_ok(self):
        good = ('[{"instruction": "材料中的第五项措施是什么?",'
                ' "input": "",'
                ' "output": "第五项措施与能源政策相关。"}]')
        gen = FakeGenerator([good])
        ic = InstructionConverter(llm_generator=gen, llm_max_attempts=3)
        ic._generate_with_llm(LIST_TEXT, make_result(LIST_TEXT))
        self.assertEqual(gen.call_count, 1)      # 一次成功, 零额外成本

    def test_gives_up_after_max_attempts(self):
        gen = FakeGenerator(["[]", "[]", "[]", "[]"])
        ic = InstructionConverter(llm_generator=gen, llm_max_attempts=3)
        samples = ic._generate_with_llm(LIST_TEXT, make_result(LIST_TEXT))
        self.assertEqual(gen.call_count, 3)      # 达上限即放弃
        self.assertEqual(samples, [])


class TestCOTRetry(unittest.TestCase):
    GOOD_CHAIN = (
        "1. 材料按编号逐条给出了五项措施。\n"
        "2. 逐条核对第一条到第五条, 共5个条目。\n"
        "因此结论是: 共5条。"
    )

    # 无数字、句末完整的链: 数字一致性与截断校验都能过(重试只由"短。"触发)
    SAFE_CHAIN = ("1. 依据材料中列出的条目逐条展开，每一步都能在材料中找到出处。\n"
                  "2. 依次核对材料中的对应条目，保证中间结论与材料表述一致且可复核。\n"
                  "因此结论是: 见材料。")

    @staticmethod
    def _calls_for(gen, cls):
        """按类统计调用次数: 四类配额通道下, 一次生成会对四类分别调用"""
        return sum(1 for p in getattr(gen, "prompts", []) if ("任务类别：%s" % cls) in p)

    def test_reasoning_retries_on_bad_chain(self):
        # 队列顺序 解释→建议→比较→推理: 前三次各返回"短。"(非推理类不重试),
        # 推理类首败后重试 → 第二次合格停止
        gen = FakeGenerator(["短。"] * 4 + [self.SAFE_CHAIN])
        cc = COTConverter(llm_generator=gen, llm_max_attempts=3)
        samples = cc._generate_cot_with_llm(LIST_TEXT, make_result(LIST_TEXT), task_type="推理")
        self.assertEqual(self._calls_for(gen, "推理"), 2)          # 首败→重试
        self.assertTrue(any(s["task_type"] == "推理" for s in samples))

    def test_non_reasoning_no_retry(self):
        # 建议类: 首次失败 → 不重试, 单次调用
        gen = FakeGenerator(["短。"])
        cc = COTConverter(llm_generator=gen, llm_max_attempts=3)
        samples = cc._generate_cot_with_llm(LIST_TEXT, make_result(LIST_TEXT), task_type="建议")
        self.assertEqual(self._calls_for(gen, "建议"), 1)          # 不重试
        self.assertEqual([s for s in samples if s["task_type"] == "建议"], [])

    def test_reasoning_gives_up_after_max_attempts(self):
        gen = FakeGenerator(["短。"])
        cc = COTConverter(llm_generator=gen, llm_max_attempts=3)
        samples = cc._generate_cot_with_llm(LIST_TEXT, make_result(LIST_TEXT), task_type="推理")
        self.assertEqual(self._calls_for(gen, "推理"), 3)          # 耗尽重试
        self.assertFalse(any(s["task_type"] == "推理" for s in samples))




class TestCOTExtractiveSeed(unittest.TestCase):
    """模板未命中(英文散文/无结构材料)时按 skill 抽取式兜底生成问答对。

    回归点: 旧实现把 COT 的 LLM 通道锁在中文模板种子上, 种子为空即一次都不调用
    → 英文扫描件/散文整份零产出(实测 11 份)。
    """

    EN_PROSE = (
        "Meteorology is the study of the atmosphere and its phenomena. "
        "Weather forecasts are produced by numerical models that solve the equations "
        "of motion and thermodynamics. Observations from surface stations, radiosondes "
        "and satellites feed these models. Forecasters must interpret the output "
        "carefully and they should compare the guidance with local observations."
    )
    EN_CHAIN = (
        "1. The material defines meteorology as the study of the atmosphere and its phenomena.\n"
        "2. It states that forecasts come from numerical models fed by observations from "
        "surface stations, radiosondes and satellites.\n"
        "3. It advises that forecasters interpret the output carefully and compare the "
        "guidance with local observations.\n"
        "因此结论是: Meteorology is the study of the atmosphere and its phenomena."
    )

    def test_english_prose_has_no_template_seed(self):
        ic = InstructionConverter(max_samples=5)
        seeds = ic._generate_from_templates(self.EN_PROSE, make_result(self.EN_PROSE))
        self.assertEqual(seeds, [], "前提: 该英文散文无模板命中")

    def test_llm_channel_still_runs_without_template_seed(self):
        # 模板为空时必须仍能生成: 四类配额通道会分别调用(样本类别取自各自的目标类别)
        gen = FakeGenerator([self.EN_CHAIN])
        cc = COTConverter(llm_generator=gen, llm_max_attempts=1)
        samples = cc._generate_cot_with_llm(self.EN_PROSE, make_result(self.EN_PROSE),
                                            task_type="建议")
        self.assertGreaterEqual(gen.call_count, 1, "模板为空时必须仍调用 LLM(抽取式兜底)")
        self.assertGreaterEqual(len(samples), 1, "英文散文应能产出 COT 样本")
        for s0 in samples:
            for k in ("instruction", "input", "chain_of_thought", "output", "task_type"):
                self.assertIn(k, s0)
            self.assertIn(s0["task_type"], ("解释", "推理", "建议", "比较"))
            self.assertTrue(s0["input"], "抽取式种子的 input 必须是材料节选")

    def test_seed_is_english_for_english_material(self):
        cc = COTConverter()
        seed = cc._extractive_seed(self.EN_PROSE, "en", "建议")
        self.assertIn("material", seed["instruction"])
        seed_zh = cc._extractive_seed(
            "本方案要求各地区加强统筹协调，落实重点任务，健全工作机制，确保各项措施落地见效。",
            "zh", "建议")
        self.assertIn("材料", seed_zh["instruction"])

    def test_chinese_unstructured_text_also_seeded(self):
        # 中文无结构散文(无标题/编号)同样走兜底
        txt = ("相关机制的原理在于热量传导，随着温差增大传导速度提升，"
               "热量会从高温区域向低温区域持续流动。")
        gen = FakeGenerator([self.EN_CHAIN])
        cc = COTConverter(llm_generator=gen, llm_max_attempts=1)
        cc._generate_cot_with_llm(txt, make_result(txt), task_type="解释")
        self.assertGreaterEqual(gen.call_count, 1)


if __name__ == "__main__":
    unittest.main()


class TestInstructionVolumeBudget(unittest.TestCase):
    """按体量生成: 不再固定只处理前 5 块(块数/条数由体量决定)"""

    GOOD = ('[{"instruction": "材料中反复出现的句子说明了什么?",'
            ' "input": "", "output": "该句用于说明材料的分块构成。"}]')

    def test_source_budget_by_volume(self):
        ic = InstructionConverter()
        self.assertEqual(ic._source_budget("字" * 1500), 3)        # 体量小 → 下限 3
        self.assertEqual(ic._source_budget("字" * 15000), 10)      # 15000/1500
        self.assertEqual(ic._source_budget("字" * 300000), 40)     # 封顶 40

    def test_all_chunks_processed_not_just_five(self):
        # 16800 字 ≈ 11 块: 旧逻辑只处理前 5 块(最多 5 次调用), 新逻辑按体量处理全部
        text = "这是用于分块测试的句子，内容长度足够构成材料。" * 800
        gen = FakeGenerator([self.GOOD])
        ic = InstructionConverter(llm_generator=gen, llm_max_attempts=3)
        samples = ic._generate_with_llm(text, make_result(text))
        self.assertGreater(gen.call_count, 5)                      # 5 块上限已取消
        self.assertGreater(len(samples), 5)                        # 产出也随体量增加
        self.assertLessEqual(len(samples), 40)                     # 仍受上限约束

    def test_stops_at_budget(self):
        # 体量远大于上限: 产出不应超过 40 条(不无限膨胀)
        text = "这是用于分块测试的句子，内容长度足够构成材料。" * 4000
        gen = FakeGenerator([self.GOOD])
        ic = InstructionConverter(llm_generator=gen, llm_max_attempts=3)
        samples = ic._generate_with_llm(text, make_result(text))
        self.assertLessEqual(len(samples), 40)
