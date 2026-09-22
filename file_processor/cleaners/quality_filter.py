"""
质量过滤器
根据多种启发式规则过滤低质量文本
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from ..utils.lang_detect import detect_language
from ..utils.dedup import Deduplicator

# "正常文本"允许出现的字符范围：ASCII + 中日韩标点 + CJK 扩展A + CJK 统一表意 +
# CJK 兼容表意 + 全角字符 + 通用标点（引号/破折号等）
_TEXT_CHAR_RANGES = (
    r'\x20-\x7e'      # ASCII 可打印
    r'\u2010-\u2027'  # 通用标点 - “ ” ‘ ’ 等
    r'\u2030-\u205e'  # 千分号、省略号、箭头等
    r'\u3000-\u303f'  # 中日韩标点　、。〈〉《》「」等
    r'\u3400-\u4dbf'  # CJK 扩展 A
    r'\u4e00-\u9fff'  # CJK 统一表意
    r'\uf900-\ufaff'  # CJK 兼容表意
    r'\uff00-\uffef'  # 全角字符
)
_ALLOWED_RE = re.compile(f'[{_TEXT_CHAR_RANGES}\\n\\r\\t]')
_DISALLOWED_RE = re.compile(f'[^{_TEXT_CHAR_RANGES}\\n\\r\\t]')

# 连续重复字符（退化输出特征，如 "啊啊啊啊啊啊啊啊…"、"。。"*N）
_LONG_RUN_RE = re.compile(r'([^\s])\1{20,}')


@dataclass
class QualityResult:
    passed: bool
    reason: Optional[str] = None


class QualityFilter:
    """质量过滤器"""

    def __init__(self,
                 min_length: int = 30,
                 max_length: int = 1_000_000,
                 garbage_ratio: float = 0.3,
                 target_languages: Optional[List[str]] = None,
                 dedup_threshold: float = 0.85,
                 keep_unknown_language: bool = True,
                 max_repeat_char_ratio: float = 0.3):
        self.min_length = min_length
        self.max_length = max_length
        self.garbage_ratio = garbage_ratio
        self.target_languages = target_languages or ["zh", "en"]
        self.keep_unknown_language = keep_unknown_language
        self.max_repeat_char_ratio = max_repeat_char_ratio
        self.dedup = Deduplicator(threshold=dedup_threshold)

    def check(self, text: str) -> QualityResult:
        """综合质量检查"""
        if not text or not text.strip():
            return QualityResult(False, "empty_text")

        # 长度检查
        if len(text) < self.min_length:
            return QualityResult(False, f"too_short: {len(text)} < {self.min_length}")
        if len(text) > self.max_length:
            return QualityResult(False, f"too_long: {len(text)} > {self.max_length}")

        # 信息密度检查
        if not self._has_minimum_information(text):
            return QualityResult(False, "insufficient_information")

        # 乱码检查
        if self._is_garbage(text):
            return QualityResult(False, "garbage_content")

        # 语言检查
        lang = detect_language(text)
        if lang == "unknown" and not self.keep_unknown_language:
            return QualityResult(False, "unknown_language")

        # 重复检查
        if self.dedup.is_duplicate(text):
            return QualityResult(False, "duplicate_content")

        return QualityResult(True)

    def _has_minimum_information(self, text: str) -> bool:
        """最小信息密度检查：正常字符占比"""
        sample = text[:500]
        ratio = len(_ALLOWED_RE.findall(sample)) / max(len(sample), 1)
        return ratio > 0.4

    def _is_garbage(self, text: str) -> bool:
        """乱码文本检测

        注意：不能使用"某个字符出现超过 N 次"这类绝对阈值 —— 中文里
        "的/国/是" 等高频字在 2000 字里出现 100+ 次完全正常，旧规则会
        把正常中文文档误判为乱码并整体丢弃。这里改用比例 + 连续重复判定。
        """
        sample = text[:2000]

        # 1) 连续重复字符（真正的退化输出）
        if _LONG_RUN_RE.search(sample):
            return True

        # 2) 单字符占比过高（正常中文最高频字占比约 5%）
        if len(sample) > 200:
            counter = Counter(ch for ch in sample if not ch.isspace())
            if counter:
                _, top_count = counter.most_common(1)[0]
                if top_count / len(sample) > self.max_repeat_char_ratio:
                    return True

        # 3) 非法/不可打印字符比例过高
        ratio = len(_DISALLOWED_RE.findall(sample)) / max(len(sample), 1)
        return ratio > self.garbage_ratio

    def reset(self):
        self.dedup.reset()