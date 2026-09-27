"""生成阶段"来源级并发"的行为契约

并发只许改"等多久", 不许改"生成什么":
  ① 真并发: 多份语料同时在飞(可观测在飞峰值)
  ② 内容一致: 并发产出的样本集合与串行完全相同
  ③ 失败隔离: 某一份抛异常不影响其他份, 且失败份不写入输出(留给 resume 重跑)
  ④ resume 不重复: 并发写盘串行化后, 去重指纹仍正确

注意: 用例语料必须彼此"不像" —— 近重复会被清洗阶段判 duplicate_content 而合并,
那样只剩 1 份可跑, 并发也就测不出来了。
"""
import json
import os
import threading
import time
import unittest

from file_processor.config import ProcessorConfig
from file_processor.pipeline import FileProcessorPipeline
from tests._tempdir import cleanup, workspace_tempdir

DOCS = [
    "机器学习是人工智能的一个重要分支，它通过数据训练模型并不断优化参数，"
    "从而在自然语言处理、计算机视觉与语音识别等领域取得显著进展。\n\n"
    "深度学习依赖多层神经网络自动提取特征，减少了人工设计特征的繁琐过程，"
    "并在大规模数据上表现出更强的泛化能力。\n\n"
    "实践中数据质量往往比模型规模更能决定最终效果，因此需要建立采集、清洗、"
    "标注与验收的完整流程，并对每一批语料留存来源与版本记录。\n",

    "今年的粮食生产形势总体平稳，夏粮收购价格保持合理区间，仓储与物流衔接顺畅。\n\n"
    "各地要继续落实耕地保护责任，稳定播种面积，推广节水灌溉与绿色防控技术，"
    "并把农机检修、烘干储运等环节的服务保障做实做细。\n\n"
    "对受灾地区要加快理赔进度，配套安排补种改种的种子与化肥供应，"
    "确保农户收益不受明显影响。\n",

    "气象观测站网的布局需要考虑地形遮挡与资料代表性，山区站点密度通常高于平原。\n\n"
    "卫星遥感反演的地表温度需与地面自动站观测做交叉校验，"
    "订正方法包括线性回归、变分同化与机器学习残差修正三类。\n\n"
    "业务上更关注降水与大风两类要素的短临预报准确率，"
    "检验指标常用命中率、虚警率与临界成功指数。\n",

    "开源许可证大致分为宽松型与传染型两类，前者允许闭源再分发，后者要求衍生作品沿用同一许可。\n\n"
    "企业在引入第三方组件时应建立清单，逐项核对其许可条款与专利授权声明，"
    "避免把传染型许可代码混入自有闭源产品。\n\n"
    "合规审查还应覆盖间接依赖：包管理器解析出的传递依赖同样受许可约束。\n",
]


class StubGenerator:
    """冒充 LLM 生成器: 不联网, 可观测并发峰值, 可注入失败"""

    def __init__(self, delay=0.2, fail_keywords=()):
        self.delay = delay
        self.fail_keywords = tuple(fail_keywords)
        self.lock = threading.Lock()
        self.in_flight = 0
        self.peak = 0
        self.calls = 0

    def generate(self, prompt, system_prompt=""):
        with self.lock:
            self.in_flight += 1
            self.calls += 1
            self.peak = max(self.peak, self.in_flight)
        try:
            if any(k in prompt for k in self.fail_keywords):
                raise RuntimeError("stub failure")
            time.sleep(self.delay)
            return json.dumps([{
                "instruction": "请概括以下材料的要点",
                "input": prompt[:200],
                "output": "材料说明了相关工作的总体安排与实施要点。",
            }], ensure_ascii=False)
        finally:
            with self.lock:
                self.in_flight -= 1


class GenerationConcurrencyTest(unittest.TestCase):

    def _build(self, workers=4, resume=False, tmp=None, fail_keywords=(), subdir="case"):
        if tmp is None:
            tmp = workspace_tempdir()
            self.addCleanup(cleanup, tmp)
        txt_dir = os.path.join(tmp, "txt")
        out_dir = os.path.join(tmp, subdir, "out")
        os.makedirs(txt_dir, exist_ok=True)
        names = []
        for i, doc in enumerate(DOCS):
            name = f"doc_{i}.txt"
            names.append(name)
            with open(os.path.join(txt_dir, name), "w", encoding="utf-8") as f:
                f.write(doc)
        cfg = ProcessorConfig()
        cfg.input_dirs = {"txt": txt_dir}
        cfg.output_dir = out_dir
        cfg.cache_dir = os.path.join(tmp, subdir, "cache")
        cfg.output_types = ["instruction"]
        cfg.enable_llm = True
        cfg.quality_enabled = False
        cfg.enable_ppl_filter = False
        cfg.sample_check_ratio = 0          # 关掉语义抽检, 只测生成阶段的并发
        cfg.llm_max_concurrency = workers
        cfg.max_workers = 2
        cfg.resume = resume
        pipeline = FileProcessorPipeline(cfg)
        stub = StubGenerator(fail_keywords=fail_keywords)
        pipeline.instruction_converter.llm_generator = stub   # 绕开真实 API
        return pipeline, stub, names, tmp

    @staticmethod
    def _samples(out_dir):
        path = os.path.join(out_dir, "instruction", "data.jsonl")
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def test_sources_run_concurrently(self):
        p, stub, _, _ = self._build(workers=4)
        p.run()
        self.assertGreaterEqual(stub.peak, 2, "应有多份语料同时在飞")
        self.assertLessEqual(stub.peak, 4, "在飞数不得超过配置路数")
        self.assertEqual(len({s["source"] for s in self._samples(p.config.output_dir)}), 4)

    def test_single_worker_stays_serial(self):
        p, stub, _, _ = self._build(workers=1)
        p.run()
        self.assertEqual(stub.peak, 1, "workers=1 时必须保持串行")

    def test_concurrent_output_equals_serial_output(self):
        serial, _, _, _ = self._build(workers=1, subdir="serial")
        serial.run()
        conc, _, _, _ = self._build(workers=4, subdir="conc")
        conc.run()

        def key(rows):
            # id 由写入层按**写入序**分配(并发完成先后不同 -> 编号不同) ->
            # 集合比对忽略 id, 只比样本内容; id 的连续唯一性单独断言
            stripped = [{k: v for k, v in r.items() if k != "id"} for r in rows]
            return sorted(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in stripped)

        self.assertTrue(self._samples(serial.config.output_dir), "串行跑应有产出")
        self.assertEqual(key(self._samples(serial.config.output_dir)),
                         key(self._samples(conc.config.output_dir)),
                         "并发产出的样本集合必须与串行完全一致")

        rows = self._samples(conc.config.output_dir)
        ids = sorted(r["id"] for r in rows if r.get("id"))
        self.assertEqual(ids, ["sft-%06d" % i for i in range(1, len(ids) + 1)],
                         "id 必须连续且唯一(缺失/sft- 前缀错都会在此暴露)")

    def test_failure_of_one_source_does_not_block_others(self):
        p, stub, _, _ = self._build(workers=4, fail_keywords=("开源许可证",))
        p.run()
        written = {s["source"] for s in self._samples(p.config.output_dir)}
        self.assertGreaterEqual(len(written), 2, "其余来源必须照常产出")
        self.assertIn("doc_0.txt", written)
        self.assertNotIn("doc_3.txt", written, "失败的来源不得写入输出(resume 会补)")
        # 转换器对 LLM 失败是逐块 try/except(不冒泡) → 该份表现为"产 0 样本",
        # 由 generation_empty 计量; 硬异常则记 generation_failures。二者必有其一。
        self.assertGreaterEqual(p.stats.generation_failures + p.stats.generation_empty, 1)
        with open(os.path.join(p.config.output_dir, "run_report.json"), encoding="utf-8") as f:
            report = json.load(f)
        self.assertGreaterEqual(report["generation_failures"] + report["generation_empty"], 1)

    def test_resume_after_concurrent_run_adds_no_duplicates(self):
        p, _, _, tmp = self._build(workers=4)
        p.run()
        before = self._samples(p.config.output_dir)
        self.assertTrue(before)
        again, _, _, _ = self._build(workers=4, resume=True, tmp=tmp)
        again.run()
        after = self._samples(again.config.output_dir)
        self.assertEqual(len(before), len(after), "resume 重跑不得重复写已产出的样本")
        self.assertGreater(again.files_skipped_resume, 0, "resume 应跳过已产出来源")


if __name__ == "__main__":
    unittest.main(verbosity=2)