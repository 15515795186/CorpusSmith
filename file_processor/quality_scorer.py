"""
LLM 质量评分器
对清洗后的文本进行多维度评分，只有通过严格阈值的数据才进入数据集转换
"""

import json
import os
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 长文本抽样标记：质检只看 开头/中部/结尾 三段，避免旧实现 text[:3000]
# 把长文在句子中间切断后、模型又按规则给"内容完整性"打低分的自伤问题
# （实测 250 篇通过率因此只有 8.1%，误杀原因清一色"结尾被截断"）。
SAMPLE_MARK = "【……中间内容因长度限制省略……】"

QUALITY_PROMPT = """你是一个严格的数据质量评估专家。请对以下文本进行多维度评分。

关于送评文本的形态：
- 长文档以抽样形式提供：开头、中部、结尾三段，段间以标记
  "【……中间内容因长度限制省略……】" 连接。
- 该省略标记是本系统插入的，不代表文档本身有缺失，不得因此扣分。
- "内容完整性"只依据两点判断：开头段是否从正常位置开始；
  结尾段是否自然收束（以完整句子/章节结束）。
- 仅当结尾段本身停在句子中间（且其后没有省略标记）时，才属于真实截断。

评分标准（每项1-5分）：
1. 内容完整性 (Content Completeness)：文本有完整的开头和结尾，没有被截断或缺失关键部分
2. 语言规范性 (Language Quality)：无乱码字符、无OCR识别错误、语法正确、标点规范
3. 信息密度 (Information Density)：包含实质性信息，不是空泛重复或模板化内容
4. 结构清晰度 (Structure Clarity)：段落层次分明，标题/列表/表格等结构清晰
5. 总体可读性 (Readability)：语言流畅，逻辑连贯，人读起来自然不费力

评分指南：
- 5分 = 极好，无需任何修改
- 4分 = 良好，偶有小瑕疵
- 3分 = 及格，有明显问题但不影响理解
- 2分 = 较差，问题严重，影响理解
- 1分 = 极差，几乎不可用

关键规则：
- 如果文本包含大量乱码、重复字符、或明显是OCR失败产物，第2项(语言规范性)必须≤2分
- 如果文本过短(<50字)或内容空洞，第3项(信息密度)必须≤2分
- 如果结尾段在句子中间突兀中断（非省略标记所致），第1项(内容完整性)必须≤2分
- 若文本以省略标记结束，说明结尾未展示，内容完整性不因抽样而扣分

请严格评分，宁低勿高。低质量的训练数据会损害模型性能。

输出格式（仅输出JSON，不要其他内容）：
```json
{
  "content_completeness": <1-5>,
  "language_quality": <1-5>,
  "information_density": <1-5>,
  "structure_clarity": <1-5>,
  "readability": <1-5>,
  "total_score": <总和>,
  "passed": <true/false>,
  "issues": ["问题1", "问题2", ...]
}
```"""


@dataclass
class QualityScore:
    """单条文本的质量评分结果"""
    content_completeness: int = 0
    language_quality: int = 0
    information_density: int = 0
    structure_clarity: int = 0
    readability: int = 0
    total_score: int = 0
    passed: bool = False
    issues: List[str] = field(default_factory=list)
    error: Optional[str] = None


class QualityScorer:
    """LLM质量评分器"""

    def __init__(self, api_key: str = "", model: str = "deepseek-flash",
                 min_per_dimension: int = 3, min_total: int = 20,
                 batch_size: int = 5, max_retries: int = 4,
                 max_workers: int = 4,
                 provider: str = "deepseek", base_url: str = ""):
        # 与文本生成同一套 provider 解析: DeepSeek 走 https://api.deepseek.com/
        env_key = "DEEPSEEK_API_KEY" if provider == "deepseek" else "DASHSCOPE_API_KEY"
        default_base = ("https://api.deepseek.com/" if provider == "deepseek"
                        else "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.api_key = api_key or os.getenv(env_key, "")
        self.provider = provider
        self.base_url = base_url or default_base
        self.model = model
        self.min_per_dimension = min_per_dimension
        self.min_total = min_total
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.max_workers = max(1, max_workers)

    @staticmethod
    def _build_sample(text: str, chunk: int = 1200) -> str:
        """长文本取 开头/中部/结尾 三段抽样；短文本原样返回

        旧实现只送 text[:3000]，长文在句中被切断，模型按"截断"规则给
        完整性打低分 → 通过率被系统性压低（实测 8.1%），属自伤误判。
        """
        if len(text) <= chunk * 3 + 100:
            return text
        head = text[:chunk]
        mid_start = (len(text) - chunk) // 2
        mid = text[mid_start:mid_start + chunk]
        tail = text[-chunk:]
        sep = f"\n\n{SAMPLE_MARK}\n\n"
        return head + sep + mid + sep + tail

    def score(self, text: str) -> QualityScore:
        """对单条文本进行质量评分"""
        if not text or len(text.strip()) < 20:
            return QualityScore(
                passed=False, issues=["文本过短或为空"],
                error="text_too_short"
            )

        try:
            from openai import OpenAI
        except ImportError as e:
            return QualityScore(passed=False, error=f"openai 未安装: {e}")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                        timeout=120, max_retries=0)
        sample = self._build_sample(text)
        last_err = ""

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": QUALITY_PROMPT},
                        {"role": "user", "content": f"请评估以下文本质量：\n\n{sample}"},
                    ],
                    temperature=0.1,
                    max_tokens=3072,     # DeepSeek 推理型模型: 扣除隐藏思考后仍要留足 JSON
                )
                content = response.choices[0].message.content
                result = self._parse_response(content)
                if result:
                    return result

                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                last_err = str(e)[:160]
                logger.warning("Quality scoring attempt %d/%d error: %s",
                               attempt + 1, self.max_retries, e)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return QualityScore(passed=False, error=last_err)

        return QualityScore(passed=False,
                            error=f"Max retries exceeded{': ' + last_err if last_err else ''}")

    def _parse_response(self, content: str) -> Optional[QualityScore]:
        """解析LLM返回的JSON评分"""
        try:
            # 提取 JSON
            json_match = re.search(r'\{[^{}]*"content_completeness"[^{}]*\}', content, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{[^{}]*"passed"[^{}]*\}', content, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)

            if not json_match:
                return None

            data = json.loads(json_match.group())

            score = QualityScore(
                content_completeness=int(data.get("content_completeness", 0)),
                language_quality=int(data.get("language_quality", 0)),
                information_density=int(data.get("information_density", 0)),
                structure_clarity=int(data.get("structure_clarity", 0)),
                readability=int(data.get("readability", 0)),
                issues=data.get("issues", []),
            )
            score.total_score = sum([
                score.content_completeness,
                score.language_quality,
                score.information_density,
                score.structure_clarity,
                score.readability,
            ])
            score.passed = self._is_passed(score)
            return score
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug("Failed to parse quality score: %s", e)
            return None

    def _is_passed(self, score: QualityScore) -> bool:
        """检查是否通过质量阈值"""
        scores = [
            score.content_completeness,
            score.language_quality,
            score.information_density,
            score.structure_clarity,
            score.readability,
        ]
        # 任何单项低于最低分 → 不通过
        if any(s < self.min_per_dimension for s in scores):
            return False
        # 总分低于最低总分 → 不通过
        if score.total_score < self.min_total:
            return False
        return True

    def score_batch(self, texts: List[str], sources: List[str],
                    progress_callback=None) -> List[Tuple[str, QualityScore, str]]:
        """并发批量评分，返回 [(text, score, source), ...]（顺序与输入一致）"""
        total = len(texts)
        if total == 0:
            return []

        logger.info("质量评分开始: %d 条文本 (并发 %d)", total, self.max_workers)
        results: List[Optional[Tuple[str, QualityScore, str]]] = [None] * total
        done = 0

        def work(idx: int):
            return idx, (texts[idx], self.score(texts[idx]), sources[idx])

        with ThreadPoolExecutor(max_workers=min(self.max_workers, total)) as executor:
            for idx, item in executor.map(work, range(total)):
                results[idx] = item
                done += 1
                if progress_callback is not None:
                    progress_callback(done, total, item[1])

        passed = sum(1 for _, s, _ in results if s.passed)
        failed = total - passed
        logger.info("质量评分完成: %d 通过, %d 淘汰 (通过率 %.1f%%)",
                     passed, failed, passed / total * 100 if total else 0)
        return [r for r in results if r is not None]

    def generate_report(self, results: List[Tuple[str, QualityScore, str]],
                        output_path: str):
        """生成质量报告"""
        from collections import Counter

        total = len(results)
        passed = sum(1 for _, s, _ in results if s.passed)
        failed = total - passed

        # 统计各维度平均分
        dims = {
            "content_completeness": [],
            "language_quality": [],
            "information_density": [],
            "structure_clarity": [],
            "readability": [],
        }
        failure_reasons = Counter()
        failed_sources: List[Dict] = []
        api_errors = 0

        for _, score, source in results:
            for dim in dims:
                dims[dim].append(getattr(score, dim))
            if score.error:
                api_errors += 1
            if not score.passed:
                for issue in score.issues:
                    failure_reasons[issue] += 1
                failed_sources.append({
                    "source": source,
                    "total_score": score.total_score,
                    "error": score.error,
                    "issues": score.issues[:3],
                })

        report = {
            "total_samples": total,
            "passed": passed,
            "failed": failed,
            "api_errors": api_errors,
            "pass_rate_percent": round(passed / total * 100, 1) if total else 0,
            "avg_dimension_scores": {
                "内容完整性": round(sum(dims["content_completeness"]) / total, 2) if total else 0,
                "语言规范性": round(sum(dims["language_quality"]) / total, 2) if total else 0,
                "信息密度": round(sum(dims["information_density"]) / total, 2) if total else 0,
                "结构清晰度": round(sum(dims["structure_clarity"]) / total, 2) if total else 0,
                "总体可读性": round(sum(dims["readability"]) / total, 2) if total else 0,
            },
            "threshold_min_per_dimension": self.min_per_dimension,
            "threshold_min_total": self.min_total,
            "model": self.model,
            "top_failure_reasons": failure_reasons.most_common(10),
            "failed_sources": failed_sources[:100],
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info("质量报告已生成: %s", output_path)
        return report