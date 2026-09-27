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


class TestCOTRetryAndDrop(unittest.TestCase):
    """COT 拒绝采样: 校验不过重试 llm_max_attempts 次, 仍不过 -> **丢弃**

    素材 = 既有 SFT 问答对(convert_sft_batch), 不再有按篇抽取的兜底问答对。
    """

    SFT = {"id": "sft-000001",
           "instruction": "请对比 ClickHouse 与 Loki 的查询能力差异。",
           "input": "ClickHouse 查询能力强；Loki 查询较弱。",
           "output": "ClickHouse 更强。",
           "source": "doc.txt"}
    GOOD_CHAIN = ("1. 材料给出两个对象各自的强弱项，逐项核对后落在同一结论上。\n"
                  "2. 综合以上事实收敛到回答。\n因此结论是: ClickHouse 更强。")

    def _cc(self, responses, attempts=3):
        gen = FakeGenerator(responses)
        return COTConverter(llm_generator=gen, llm_max_attempts=attempts), gen

    def test_no_retry_when_first_ok(self):
        cc, gen = self._cc([self.GOOD_CHAIN])
        samples = cc.convert_sft_batch("doc.txt", [self.SFT], corpus_text=self.SFT["input"])
        self.assertEqual(gen.call_count, 1, "一次成功, 零额外成本")
        self.assertEqual(len(samples), 1)

    def test_retry_on_bad_chain(self):
        cc, gen = self._cc(["短。", self.GOOD_CHAIN])
        samples = cc.convert_sft_batch("doc.txt", [self.SFT], corpus_text=self.SFT["input"])
        self.assertEqual(gen.call_count, 2, "首次过短 -> 重试")
        self.assertEqual(len(samples), 1)

    def test_gives_up_and_drops_after_max_attempts(self):
        cc, gen = self._cc(["短。"] * 5)
        samples = cc.convert_sft_batch("doc.txt", [self.SFT], corpus_text=self.SFT["input"])
        self.assertEqual(gen.call_count, 3, "达上限即放弃")
        self.assertEqual(samples, [])
        self.assertEqual(cc.stats["dropped"], 1)


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
