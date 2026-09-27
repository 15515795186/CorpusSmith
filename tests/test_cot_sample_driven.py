"""COT 阶段(样本驱动)端到端契约

素材必须来自既有 SFT 问答对, 且:
  ① 判类只在 SFT 样本层, 四类之外不进 COT
  ② 生成记录带规范字段(id / verifiable / final_answer_correct / reasoning_correct / source)
  ③ source.sft_id 指回真实 SFT 样本, 三个问答字段与 SFT 逐字一致(素材不得自造)
  ④ 判类与丢弃统计写进 metadata(便于人工核对素材层")
"""
import json
import os
import re
import unittest

from file_processor.config import ProcessorConfig
from file_processor.pipeline import FileProcessorPipeline
from tests._tempdir import cleanup, workspace_tempdir

DOC = (
    "甲方案与乙方案的成本与见效速度差异\n\n"
    "甲方案成本低但见效慢，乙方案成本高但见效快，选型时要结合预算与工期权衡。\n"
)


class CotStubGenerator:
    """冒充 LLM: SFT 阶段返回一条四类(比较)问答对; COT 阶段返回合格思维链"""

    SFT_SAMPLE = {
        "instruction": "请对比甲方案与乙方案在成本与见效速度上的差异。",
        "input": "甲方案成本低但见效慢；乙方案成本高但见效快。",
        "output": "甲方案更省钱，乙方案更快见效。",
    }
    CHAIN = ("1. 材料给出两个方案在成本与见效速度上的差异，逐项核对。\n"
             "2. 综合以上事实收敛到回答。\n因此结论是: 甲方案更省钱，乙方案更快见效。")

    def __init__(self):
        self.sft_calls = 0
        self.cot_calls = 0
        self.cot_prompts = []

    def generate(self, prompt, system_prompt=""):
        if "chain-of-thought" in prompt:
            self.cot_calls += 1
            self.cot_prompts.append(prompt)
            return self.CHAIN
        self.sft_calls += 1
        return json.dumps([self.SFT_SAMPLE], ensure_ascii=False)


class CotSampleDrivenPipelineTest(unittest.TestCase):

    def setUp(self):
        self._tmp = workspace_tempdir()
        self.addCleanup(cleanup, self._tmp)
        txt_dir = os.path.join(self._tmp, "txt")
        os.makedirs(txt_dir, exist_ok=True)
        with open(os.path.join(txt_dir, "plan.txt"), "w", encoding="utf-8") as f:
            f.write(DOC)
        cfg = ProcessorConfig()
        cfg.input_dirs = {"txt": txt_dir}
        cfg.output_dir = os.path.join(self._tmp, "out")
        cfg.cache_dir = os.path.join(self._tmp, "cache")
        cfg.output_types = ["instruction", "cot"]
        cfg.enable_llm = True
        cfg.quality_enabled = False
        cfg.enable_ppl_filter = False
        cfg.sample_check_ratio = 0
        cfg.llm_max_concurrency = 1
        cfg.max_workers = 1
        cfg.review_output = os.path.join(self._tmp, "review.jsonl")
        self.cfg = cfg
        self.pipeline = FileProcessorPipeline(cfg)
        self.stub = CotStubGenerator()
        self.pipeline.instruction_converter.llm_generator = self.stub
        self.pipeline.cot_converter.llm_generator = self.stub

    def _rows(self, name):
        path = os.path.join(self.cfg.output_dir, name, "data.jsonl")
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_cot_derives_from_sft_samples(self):
        self.pipeline.run()
        sft = self._rows("instruction")
        cot = self._rows("cot")
        self.assertTrue(sft, "SFT 阶段应有产出")
        self.assertTrue(cot, "COT 应能从 SFT 样本生成")
        self.assertGreaterEqual(self.stub.cot_calls, 1)

        s = cot[0]
        self.assertEqual(s["task_type"], "比较")
        self.assertRegex(s["id"], r"^cot-比较-\d{6}$")
        self.assertTrue(s["chain_of_thought"].strip())
        self.assertFalse(s["verifiable"], "比较类无可唯一验证答案")
        self.assertIsNone(s["final_answer_correct"], "未验证不得填")
        self.assertEqual(s["source"]["corpus_file"], "plan.txt")
        self.assertEqual(s["source"]["corpus_type"], "txt")

        # 素材不得自造 + sft_id 必须指回真实 SFT 样本
        sft_ids = {r["id"] for r in sft}
        self.assertIn(s["source"]["sft_id"], sft_ids)
        origin = [r for r in sft if r["id"] == s["source"]["sft_id"]][0]
        self.assertEqual((s["instruction"], s["input"], s["output"]),
                         (origin["instruction"], origin["input"], origin["output"]))

        # 判类统计落 metadata(素材层可核对)
        with open(os.path.join(self.cfg.output_dir, "cot", "metadata.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["status"], "completed")
        self.assertGreaterEqual(meta["stats"]["candidates"], 1)
        self.assertEqual(meta["stats"]["uncertain"], 0)

    def test_uncertain_and_out_of_scope_are_not_in_main_set(self):
        # SFT 样本改成"总结"(四类之外) -> COT 不产出, 且统计留证
        self.stub.SFT_SAMPLE = dict(self.stub.SFT_SAMPLE,
                                    instruction="请总结这份材料的主要内容。")
        self.pipeline.run()
        self.assertEqual(self._rows("cot"), [], "四类之外不得进 COT")
        with open(os.path.join(self.cfg.output_dir, "cot", "metadata.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.assertGreaterEqual(meta["stats"]["out_of_scope"], 1)
        self.assertEqual(meta["stats"]["generated"], 0)


if __name__ == "__main__":
    unittest.main()
