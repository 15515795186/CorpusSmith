"""端到端测试：不依赖任何 API key，覆盖扫描 → 读取 → 清洗 → 转换 → 输出 → resume"""

import json
import os
import unittest

from file_processor.config import ProcessorConfig
from file_processor.pipeline import FileProcessorPipeline
from file_processor.readers.base import ReaderResult
from tests._tempdir import cleanup, workspace_tempdir

TXT_A = (
    "机器学习是人工智能的一个重要分支，它通过数据训练模型并不断优化参数，"
    "从而在自然语言处理、计算机视觉与语音识别等领域取得显著进展。\n\n"
    "深度学习依赖多层神经网络自动提取特征，减少了人工设计特征的繁琐过程，"
    "并在大规模数据上表现出更强的泛化能力。\n\n"
    "1. 采集原始语料并校验来源\n"
    "2. 清洗语料并统一编码格式\n"
    "3. 标注语料并划分训练集\n"
)

TXT_SHORT = "太短"

HTML_DOC = """<html><head><title>测试页面</title></head><body>
<h1>数据结构的分类</h1>
<p>常见的数据结构包括数组、链表、栈、队列、树与图，它们在不同场景下各有优劣。</p>
<h2>线性结构</h2>
<ul>
<li>数组与链表适合顺序访问与频繁插入的场景</li>
<li>栈与队列适合具有先后顺序约束的任务</li>
<li>哈希表适合以键值对快速定位元素的场景</li>
</ul>
</body></html>"""


class PipelineEndToEndTest(unittest.TestCase):

    def setUp(self):
        self._tmp = workspace_tempdir()
        root = self._tmp
        self.txt_dir = os.path.join(root, "txt")
        self.html_dir = os.path.join(root, "html")
        self.out_dir = os.path.join(root, "out")
        os.makedirs(self.txt_dir)
        os.makedirs(self.html_dir)

        self._write(os.path.join(self.txt_dir, "a.txt"), TXT_A)
        self._write(os.path.join(self.txt_dir, "b.txt"),
                    TXT_A.replace("机器学习", "统计学习").replace("深度学习", "表示学习"))
        self._write(os.path.join(self.txt_dir, "short.txt"), TXT_SHORT)
        self._write(os.path.join(self.html_dir, "page.html"), HTML_DOC)

        self.config = ProcessorConfig()
        self.config.input_dirs = {"txt": self.txt_dir, "html_source": self.html_dir}
        self.config.output_dir = self.out_dir
        self.config.cache_dir = os.path.join(root, "cache")
        self.config.quality_enabled = False
        # PPL 是本地小 LM 的打分闸: 端到端用例用合成短文本, 不该依赖几百 MB 模型缓存
        self.config.enable_ppl_filter = False
        self.config.max_workers = 2
        self.config.chunk_size = 200
        self.config.chunk_overlap = 20
        self.config.save_rejected = True

    def tearDown(self):
        cleanup(self._tmp)

    @staticmethod
    def _write(path, content):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _count(self, name):
        path = os.path.join(self.out_dir, name, "data.jsonl")
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def test_full_run_outputs_expected_artifacts(self):
        pipeline = FileProcessorPipeline(self.config)
        pipeline.run()

        self.assertEqual(pipeline.stats.total_files, 4)
        self.assertEqual(pipeline.stats.files_processed, 4)

        # 三种数据集都要产出
        for name in ("pretrain", "instruction", "cot"):
            self.assertTrue(os.path.exists(os.path.join(self.out_dir, name, "data.jsonl")),
                            f"{name} 数据集未生成")
            self.assertGreater(self._count(name), 0, f"{name} 数据集为空")

            with open(os.path.join(self.out_dir, name, "metadata.json"), encoding="utf-8") as f:
                metadata = json.load(f)
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["records"], self._count(name))

        # 过短文本应被过滤并记录原因
        rejected_path = os.path.join(self.out_dir, "rejected.jsonl")
        self.assertTrue(os.path.exists(rejected_path))
        with open(rejected_path, encoding="utf-8") as f:
            rejected = [json.loads(line) for line in f if line.strip()]
        self.assertTrue(any(r["source"] == "short.txt" for r in rejected))
        self.assertTrue(any("too_short" in (r["reason"] or "") for r in rejected))

        # 运行报告
        with open(os.path.join(self.out_dir, "run_report.json"), encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(report["total_files"], 4)
        self.assertEqual(report["config"]["dashscope_api_key"], "")
        self.assertIn("pretrain", report["datasets"])

    def test_ppl_score_error_passes_sample_through(self):
        """PPL 单条打分异常时必须放行(不淘汰)并计数, 不能因一个坏样本炸全流程"""
        pipeline = FileProcessorPipeline(self.config)

        class RaisingFilter:          # 冒充 PPLFilter: 只有 threshold 与 check
            threshold = 13.78

            def check(self, text):
                raise RuntimeError("模拟打分异常")

        pipeline.ppl_filter = RaisingFilter()
        results = [ReaderResult(text=TXT_A, filepath=os.path.join(self.txt_dir, "a.txt"),
                                file_type="txt")]
        kept = pipeline._ppl_check(results)
        self.assertEqual(len(kept), 1, "打分异常必须放行, 不得淘汰")
        self.assertEqual(pipeline.stats.ppl_errors, 1)
        self.assertEqual(pipeline.stats.ppl_filtered, 0)

    def test_rerun_does_not_duplicate_records(self):
        FileProcessorPipeline(self.config).run()
        first = {name: self._count(name) for name in ("pretrain", "instruction", "cot")}

        FileProcessorPipeline(self.config).run()
        second = {name: self._count(name) for name in ("pretrain", "instruction", "cot")}

        self.assertEqual(first, second, "重跑必须重建输出，不能堆叠旧记录")

    def test_resume_skips_done_sources_before_scoring(self):
        FileProcessorPipeline(self.config).run()
        first = {name: self._count(name) for name in ("pretrain", "instruction", "cot")}

        resume_config = ProcessorConfig(**{
            **self.config.to_dict(),
            "resume": True,
        })
        pipeline = FileProcessorPipeline(resume_config)
        pipeline.run()
        second = {name: self._count(name) for name in ("pretrain", "instruction", "cot")}

        self.assertEqual(first, second)
        # done = {a.txt, page.html}（已产出）∪ {b.txt(近重复), short.txt(过短)}（确定性丢弃）
        # → 本轮无事可做，连读取都不发生（这正是前移跳过的意义）
        self.assertEqual(pipeline.files_skipped_resume, 4)
        self.assertEqual(pipeline.stats.cache_hits, 0)
        self.assertEqual(pipeline.stats.files_processed, 0)

    def test_resume_processes_only_new_files(self):
        pipeline = FileProcessorPipeline(self.config)
        pipeline.run()
        first = {name: self._count(name) for name in ("pretrain", "instruction", "cot")}
        base_meta = json.load(open(os.path.join(self.out_dir, "pretrain", "metadata.json"),
                                   encoding="utf-8"))
        # b.txt 是 a.txt 的近重复（过滤阶段确定性丢弃），short.txt 过短 → 均不在输出
        self.assertEqual(base_meta["sources"], 2)

        # 新主题文档，避开与 a/b 的 SimHash 相似区
        self._write(os.path.join(self.txt_dir, "c.txt"),
                    "红烧肉的做法：五花肉切块焯水，冰糖炒色后下肉翻炒，"
                    "加生抽老抽料酒与葱姜，小火焖炖四十分钟收汁即成。\n\n"
                    "清蒸鲈鱼的做法：鱼身改刀抹盐铺姜丝，水开后旺火蒸八分钟，"
                    "淋蒸鱼豉油与热油，葱丝点缀即可上桌食用。\n\n"
                    "番茄炒蛋的做法：鸡蛋打散炒至凝固盛出，番茄炒软出汁后回锅，"
                    "加糖与盐调味翻炒均匀，撒葱花装盘完成整道菜。\n")

        resume_config = ProcessorConfig(**{**self.config.to_dict(), "resume": True})
        pipeline = FileProcessorPipeline(resume_config)
        pipeline.run()

        second = {name: self._count(name) for name in ("pretrain", "instruction", "cot")}
        self.assertGreater(second["pretrain"], first["pretrain"], "新文件 c.txt 应产出新记录")

        # 本轮发现 5 个文件；done = {a,page}（已产出）∪ {b,short}（确定性丢弃）
        # → kept 只剩 c.txt
        self.assertEqual(pipeline.files_skipped_resume, 4)
        self.assertEqual(pipeline.stats.files_processed, 1)
        meta = json.load(open(os.path.join(self.out_dir, "pretrain", "metadata.json"),
                              encoding="utf-8"))
        self.assertEqual(meta["sources"], 3, "metadata.sources 必须是累计口径")
        self.assertEqual(meta["sources_this_run"], 1, "本次新增来源单独记录")
        with open(os.path.join(self.out_dir, "run_report.json"), encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(report["total_files"], 5, "发现总数 = 待处理 1 + 跳过 4")
        self.assertEqual(report["files_skipped_resume"], 4)

    def test_cache_disabled_still_works(self):
        self.config.use_read_cache = False
        pipeline = FileProcessorPipeline(self.config)
        pipeline.run()
        self.assertEqual(pipeline.stats.cache_hits, 0)
        self.assertGreater(self._count("pretrain"), 0)

    def test_scan_is_deterministic(self):
        pipeline = FileProcessorPipeline(self.config)
        first = pipeline._scan_files()
        second = FileProcessorPipeline(self.config)._scan_files()
        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first), "扫描结果应按路径排序")

    def test_missing_input_dir_is_skipped(self):
        self.config.input_dirs = {"txt": os.path.join(self._tmp, "nope")}
        pipeline = FileProcessorPipeline(self.config)
        self.assertEqual(pipeline._scan_files(), [])




class QualityApiErrorTest(unittest.TestCase):
    """评分调用失败(断网/限流/超时) ≠ 真低质淘汰

    断网只在当轮少这份语料(未评分的文本不得放行), 但要单独留证:
      ① quality_api_error.jsonl 列出待救回来源
      ② rejected.jsonl 的 stage=quality_api_error
      ③ 不进 resume 的"确定性丢弃"集合 → 重跑自动重评救回
    """

    def _pipeline(self, stub):
        tmp = workspace_tempdir()
        self.addCleanup(cleanup, tmp)
        txt_dir = os.path.join(tmp, "txt")
        os.makedirs(txt_dir)
        with open(os.path.join(txt_dir, "a.txt"), "w", encoding="utf-8") as f:
            f.write(TXT_A)
        config = ProcessorConfig()
        config.input_dirs = {"txt": txt_dir}
        config.output_dir = os.path.join(tmp, "out")
        config.cache_dir = os.path.join(tmp, "cache")
        config.quality_enabled = True
        config.enable_ppl_filter = False
        config.output_types = ["pretrain"]
        config.max_workers = 1
        p = FileProcessorPipeline(config)
        p.quality_scorer = stub          # 绕开 API: 直接注入桩
        return p

    def test_api_error_source_kept_for_retry(self):
        from file_processor.quality_scorer import QualityScore

        class StubError:
            def score_batch(self, texts, sources, progress_callback=None):
                return [(t, QualityScore(passed=False, issues=[], error="Connection error"), s)
                        for t, s in zip(texts, sources)]

            def generate_report(self, scored, path):
                return {"passed": 0, "failed": len(scored), "api_errors": len(scored),
                        "pass_rate_percent": 0.0}

        p = self._pipeline(StubError())
        p.run()

        # ① 单独留证
        api_path = os.path.join(p.config.output_dir, "quality_api_error.jsonl")
        self.assertTrue(os.path.exists(api_path), "应生成 quality_api_error.jsonl")
        with open(api_path, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["source"], "a.txt")
        self.assertIn("Connection error", recs[0]["error"])

        # ② 未评分的文本不得进数据集(零通过时输出文件不会创建)
        out_path = os.path.join(p.config.output_dir, "pretrain", "data.jsonl")
        written = []
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as f:
                written = [json.loads(l) for l in f if l.strip()]
        self.assertEqual([r["source"] for r in written if r.get("source") == "a.txt"], [])

        # ③ stage 标注 + resume 可救回
        with open(p.rejected_path, encoding="utf-8") as f:
            rejected = [json.loads(l) for l in f if l.strip()]
        stages = {r["stage"] for r in rejected}
        self.assertIn("quality_api_error", stages)
        self.assertNotIn("a.txt", p._previous_deterministic_rejections(),
                         "断网来源不得进入跳过集合, 否则重跑不救回")

        # ④ 报告计数
        with open(os.path.join(p.config.output_dir, "run_report.json"), encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(report["quality_api_errors"], 1)

    def test_low_quality_rejection_is_not_api_error(self):
        from file_processor.quality_scorer import QualityScore

        class StubLowScore:
            def score_batch(self, texts, sources, progress_callback=None):
                return [(t, QualityScore(passed=False, total_score=8, issues=["模板化表达"]), s)
                        for t, s in zip(texts, sources)]

            def generate_report(self, scored, path):
                return {"passed": 0, "failed": len(scored), "api_errors": 0,
                        "pass_rate_percent": 0.0}

        p = self._pipeline(StubLowScore())
        p.run()
        api_path = os.path.join(p.config.output_dir, "quality_api_error.jsonl")
        self.assertFalse(os.path.exists(api_path), "真低质不应写进 api 错误清单")
        with open(p.rejected_path, encoding="utf-8") as f:
            rejected = [json.loads(l) for l in f if l.strip()]
        self.assertEqual({r["stage"] for r in rejected}, {"quality"})
        self.assertNotIn("a.txt", p._previous_deterministic_rejections())




class QualityExemptionTest(unittest.TestCase):
    """人工豁免: 质检未过但经人工复核保留(典型: "结尾自然收束"判据误伤节选型来源)

    要求: ① 名单内来源放行 ② 名单外照旧淘汰 ③ 逐条留证 quality_exempted.jsonl
    ④ run_report 计数, 便于审计
    """

    def _pipeline(self, stub, exempt=None):
        tmp = workspace_tempdir()
        self.addCleanup(cleanup, tmp)
        txt_dir = os.path.join(tmp, "txt")
        os.makedirs(txt_dir)
        TXT_B = ("本市推进政务服务标准化建设，梳理高频事项清单，压减申请材料与办理时限，"
                 "推动更多事项网上办、掌上办，方便企业与群众办事创业。")
        for name, body in (("a.txt", TXT_A), ("b.txt", TXT_B)):
            with open(os.path.join(txt_dir, name), "w", encoding="utf-8") as f:
                f.write(body)
        config = ProcessorConfig()
        config.input_dirs = {"txt": txt_dir}
        config.output_dir = os.path.join(tmp, "out")
        config.cache_dir = os.path.join(tmp, "cache")
        config.quality_enabled = True
        config.enable_ppl_filter = False
        config.output_types = ["pretrain"]
        config.max_workers = 1
        if exempt:
            ex_path = os.path.join(tmp, "exempt.json")
            with open(ex_path, "w", encoding="utf-8") as f:
                json.dump({"sources": exempt}, f, ensure_ascii=False)
            config.quality_exempt_file = ex_path
        else:
            config.quality_exempt_file = os.path.join(tmp, "missing.json")
        p = FileProcessorPipeline(config)
        p.quality_scorer = stub
        return p

    def _stub(self):
        from file_processor.quality_scorer import QualityScore

        class StubAllFail:
            def score_batch(self, texts, sources, progress_callback=None):
                return [(t, QualityScore(content_completeness=2, language_quality=5,
                                         information_density=5, structure_clarity=5,
                                         readability=5, total_score=22, passed=False,
                                         issues=["内容完整性不足：结尾段突兀中断"]), s)
                        for t, s in zip(texts, sources)]

            def generate_report(self, scored, path):
                return {"passed": 0, "failed": len(scored), "api_errors": 0, "pass_rate_percent": 0.0}

        return StubAllFail()

    def test_exempted_source_passes_and_is_recorded(self):
        p = self._pipeline(self._stub(), exempt=[{"source": "a.txt", "reason": "仅结尾截断"}])
        p.run()
        ex_path = os.path.join(p.config.output_dir, "quality_exempted.jsonl")
        self.assertTrue(os.path.exists(ex_path), "应生成 quality_exempted.jsonl")
        with open(ex_path, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        self.assertEqual([r["source"] for r in recs], ["a.txt"])
        self.assertEqual(recs[0]["total_score"], 22)
        self.assertEqual(recs[0]["dims"]["content_completeness"], 2)

        # 来源名存在 meta 里, 用 formatter 的 read_sources 取(与 resume 判据同源)
        srcs = p.formatter.read_sources("pretrain")
        self.assertIn("a.txt", srcs, "豁免来源应进入数据集")
        self.assertNotIn("b.txt", srcs, "名单外来源照旧淘汰")

        report = json.load(open(os.path.join(p.config.output_dir, "run_report.json"), encoding="utf-8"))
        self.assertEqual(report["quality_exempted"], 1)


if __name__ == "__main__":
    unittest.main()