"""端到端测试：不依赖任何 API key，覆盖扫描 → 读取 → 清洗 → 转换 → 输出 → resume"""

import json
import os
import unittest

from file_processor.config import ProcessorConfig
from file_processor.pipeline import FileProcessorPipeline
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


if __name__ == "__main__":
    unittest.main()