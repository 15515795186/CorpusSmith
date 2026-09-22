"""质量评分器离线单测：抽样策略与响应解析（不发起 API 调用）"""
import unittest

from file_processor.quality_scorer import SAMPLE_MARK, QualityScorer


class BuildSampleTest(unittest.TestCase):
    """旧缺陷回归：只送 text[:3000] 导致长文在句中被切断，
    模型按"截断→完整性≤2"规则系统性误杀（实测通过率 8.1%）"""

    def test_short_text_returned_whole(self):
        text = "短文本" * 50
        self.assertEqual(QualityScorer._build_sample(text, chunk=1000), text)

    def test_long_text_keeps_head_mid_and_real_tail(self):
        text = "开头标记句子。" + "中间填充内容。" * 2000 + "结尾标记句子。"
        sample = QualityScorer._build_sample(text, chunk=1200)
        self.assertIn("开头标记句子", sample)
        self.assertIn("结尾标记句子", sample)
        self.assertTrue(sample.endswith("结尾标记句子。"),
                        "抽样必须以文档真实结尾收束，否则完整性判断失真")
        self.assertEqual(sample.count(SAMPLE_MARK), 2)

    def test_sample_bounded_size(self):
        text = "字" * 200000
        sample = QualityScorer._build_sample(text, chunk=1200)
        self.assertLess(len(sample), 4000)


class ParseResponseTest(unittest.TestCase):

    def setUp(self):
        self.scorer = QualityScorer(api_key="sk-test", min_per_dimension=3,
                                    min_total=20)

    def test_fenced_json_parsed(self):
        content = '''这是评分结果：
```json
{"content_completeness": 4, "language_quality": 4, "information_density": 5,
 "structure_clarity": 4, "readability": 4, "passed": true, "issues": []}
```'''
        score = self.scorer._parse_response(content)
        self.assertIsNotNone(score)
        self.assertEqual(score.total_score, 21)
        self.assertTrue(score.passed)

    def test_low_dimension_fails_even_if_total_high(self):
        content = ('{"content_completeness": 5, "language_quality": 1, '
                   '"information_density": 5, "structure_clarity": 5, '
                   '"readability": 5, "issues": ["乱码"]}')
        score = self.scorer._parse_response(content)
        self.assertEqual(score.total_score, 21)
        self.assertFalse(score.passed, "单项低于阈值必须不通过")

    def test_garbage_response_returns_none(self):
        self.assertIsNone(self.scorer._parse_response("我不输出JSON"))

    def test_score_batch_empty_input(self):
        self.assertEqual(self.scorer.score_batch([], []), [])


if __name__ == "__main__":
    unittest.main()