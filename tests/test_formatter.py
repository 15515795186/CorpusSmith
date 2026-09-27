import json
import os
import unittest

from file_processor.formatters.jsonl_formatter import JSONLFormatter
from tests._tempdir import cleanup, workspace_tempdir


class JSONLFormatterTest(unittest.TestCase):

    def setUp(self):
        self._tmp = workspace_tempdir()
        self.out = os.path.join(self._tmp, "datasets")

    def tearDown(self):
        cleanup(self._tmp)

    def _record(self, source, text="hello"):
        return {"text": text, "meta": {"source": source, "language": "en"}}

    def test_nested_layout_paths(self):
        formatter = JSONLFormatter(self.out)
        self.assertTrue(formatter.get_path("pretrain").endswith(
            os.path.join("pretrain", "data.jsonl")))
        self.assertTrue(formatter.get_metadata_path("cot").endswith(
            os.path.join("cot", "metadata.json")))
        self.assertTrue(formatter.get_report_path().endswith("quality_report.json"))

    def test_prepare_truncates_existing_output(self):
        formatter = JSONLFormatter(self.out)
        formatter.prepare("pretrain")
        formatter.write_batch("pretrain", [self._record("a.txt"), self._record("b.txt")])
        self.assertEqual(formatter.count_lines("pretrain"), 2)

        # 新一轮运行（默认 overwrite=True）必须清空，避免重复样本堆积
        rerun = JSONLFormatter(self.out)
        rerun.prepare("pretrain")
        self.assertEqual(rerun.count_lines("pretrain"), 0)
        rerun.write_batch("pretrain", [self._record("a.txt")])
        self.assertEqual(rerun.count_lines("pretrain"), 1)

    def test_resume_mode_keeps_existing(self):
        formatter = JSONLFormatter(self.out, overwrite=False)
        formatter.prepare("pretrain")
        formatter.write_batch("pretrain", [self._record("a.txt"), self._record("b.txt")])

        resumed = JSONLFormatter(self.out, overwrite=False)
        resumed.prepare("pretrain")
        self.assertEqual(resumed.count_lines("pretrain"), 2)
        self.assertEqual(resumed.read_sources("pretrain"), {"a.txt", "b.txt"})

    def test_read_sources_supports_flat_schema(self):
        formatter = JSONLFormatter(self.out, overwrite=False)
        formatter.prepare("instruction")
        formatter.write_batch("instruction", [{"instruction": "q", "output": "a",
                                               "source": "doc.pdf"}])
        self.assertEqual(formatter.read_sources("instruction"), {"doc.pdf"})

    def test_write_metadata(self):
        formatter = JSONLFormatter(self.out)
        formatter.prepare("cot")
        formatter.write_batch("cot", [self._record("a.txt")])
        metadata = formatter.write_metadata("cot", sources=1)
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["records"], 1)

        with open(formatter.get_metadata_path("cot"), "r", encoding="utf-8") as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk["sources"], 1)
        self.assertEqual(on_disk["dataset_name"], "cot")

    def test_prefix_is_applied(self):
        formatter = JSONLFormatter(self.out, prefix="v2_")
        self.assertTrue(formatter.get_path("pretrain").endswith("v2_data.jsonl"))

    def test_write_batch_without_prepare_is_safe(self):
        formatter = JSONLFormatter(self.out)
        formatter.write_batch("pretrain", [self._record("a.txt")])
        self.assertEqual(formatter.count_lines("pretrain"), 1)

    def test_write_batch_skips_exact_duplicates(self):
        """同一数据集内完全相同的记录只写一次（跨来源重复样本回归）"""
        formatter = JSONLFormatter(self.out)
        formatter.prepare("instruction")
        rec = self._record("a.txt")
        written = formatter.write_batch(
            "instruction", [rec, dict(rec), self._record("b.txt")])
        self.assertEqual(written, 2)
        self.assertEqual(formatter.count_lines("instruction"), 2)
        self.assertEqual(formatter.duplicates_skipped, 1)

    def test_resume_loads_existing_record_fingerprints(self):
        """续跑时历史输出中的相同记录也不会被再次写入"""
        formatter = JSONLFormatter(self.out, overwrite=False)
        formatter.prepare("pretrain")
        formatter.write_batch("pretrain", [
            self._record("a.txt", "x"), self._record("b.txt", "y")])

        resumed = JSONLFormatter(self.out, overwrite=False)
        resumed.prepare("pretrain")
        written = resumed.write_batch("pretrain", [
            self._record("a.txt", "x"),   # 与历史记录完全相同 → 跳过
            self._record("c.txt", "z"),   # 新记录 → 写入
        ])
        self.assertEqual(written, 1)
        self.assertEqual(resumed.count_lines("pretrain"), 3)


if __name__ == "__main__":
    unittest.main()


class SFTFieldEnrichTest(unittest.TestCase):
    """SFT 输出补 id/language(《数据集介绍.md》3.2): 写入层注入, 不参与去重指纹"""

    def setUp(self):
        self._tmp = workspace_tempdir()
        self.out = os.path.join(self._tmp, "datasets")

    def tearDown(self):
        cleanup(self._tmp)

    @staticmethod
    def _sft(instruction, output, source="a.txt"):
        return {"instruction": instruction, "input": "", "output": output, "source": source}

    def _rows(self, name):
        with open(os.path.join(self.out, name, "data.jsonl"), encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]

    def test_instruction_gets_id_and_language(self):
        f = JSONLFormatter(self.out)
        f.write_batch("instruction", [self._sft("请总结要点。", "本方案要求加强统筹协调。")])
        rec = self._rows("instruction")[0]
        self.assertEqual(rec["id"], "sft-000001")
        self.assertEqual(rec["language"], "zh")

    def test_ids_are_sequential_and_stable(self):
        f = JSONLFormatter(self.out)
        f.write_batch("instruction", [
            self._sft("问题一？", "答案一。"),
            self._sft("问题二？", "答案二。"),
        ])
        f.write_batch("instruction", [self._sft("问题三？", "答案三。")])
        self.assertEqual([r["id"] for r in self._rows("instruction")],
                         ["sft-000001", "sft-000002", "sft-000003"])

    def test_mixed_technames_count_as_chinese(self):
        """中文句夹英文专名应判 zh(规范 language 取 zh/en, 不留 mixed)"""
        f = JSONLFormatter(self.out)
        f.write_batch("instruction", [self._sft(
            "对比 ClickHouse、Loki 和 Elasticsearch 的查询能力差异。",
            "Loki 的查询功能较弱, Elasticsearch 支持复杂全文检索。")])
        self.assertEqual(self._rows("instruction")[0]["language"], "zh")

    def test_english_language_detected(self):
        f = JSONLFormatter(self.out)
        f.write_batch("instruction", [self._sft("Summarize the report.",
                                                "The plan requires coordinated work.")])
        self.assertEqual(self._rows("instruction")[0]["language"], "en")

    def test_pretrain_records_untouched(self):
        f = JSONLFormatter(self.out)
        rec = {"text": "hello", "meta": {"source": "a.txt"}}
        f.write_batch("pretrain", [rec])
        on_disk = self._rows("pretrain")[0]
        self.assertNotIn("id", on_disk)
        self.assertNotIn("language", on_disk)
        self.assertEqual(on_disk, rec)

    def test_duplicate_still_skipped_after_enrich(self):
        f = JSONLFormatter(self.out)
        f.write_batch("instruction", [self._sft("同一问题？", "同一答案。")])
        f.write_batch("instruction", [self._sft("同一问题？", "同一答案。")])
        self.assertEqual(len(self._rows("instruction")), 1)

    def test_resume_continues_id_sequence(self):
        first = JSONLFormatter(self.out)
        first.write_batch("instruction", [self._sft("问题一？", "答案一。")])
        resumed = JSONLFormatter(self.out, overwrite=False)
        resumed.write_batch("instruction", [self._sft("问题二？", "答案二。")])
        self.assertEqual([r["id"] for r in self._rows("instruction")],
                         ["sft-000001", "sft-000002"])
