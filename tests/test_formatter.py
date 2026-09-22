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


if __name__ == "__main__":
    unittest.main()