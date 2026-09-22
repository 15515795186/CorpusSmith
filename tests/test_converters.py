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


if __name__ == "__main__":
    unittest.main()