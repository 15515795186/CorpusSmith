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

# ---- 空白表单/网格模板检测（在 400 文件真实语料上标定：248 个已通过质检的
# 文档零误伤，精准命中空申报表/推荐表模板）----
# 叙述性文本：中文连续 8 字以上，或英文连续三个长词（表格式文档几乎为零）
_NARR_RE = re.compile(r'[\u4e00-\u9fff]{8,}|[A-Za-z]{5,}\s+[A-Za-z]{5,}\s+[A-Za-z]{5,}')


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
                 dedup_threshold: float = 0.90,
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

        # 空白表单/网格模板检查：主体是空单元格网格的申报书、推荐表、审批表
        # 学不出任何东西，且白白消耗质检 API——在启发式层就拦下
        if self._is_form_template(text):
            return QualityResult(False, "form_template_grid")

        # 重复检查
        if self.dedup.is_duplicate(text):
            return QualityResult(False, "duplicate_content")

        return QualityResult(True)

    # ---------- 空白表单检测 ----------
    _FORM_PIPE_MIN = 0.45    # 管道表格行字符占比下限
    _FORM_NARR_MAX = 0.30    # 叙述性正文占比上限
    _FORM_EMPTY_MIN = 0.50   # 空单元格占比下限（与 dup 二选一）
    _FORM_DUP_MIN = 0.40     # "复读单元格"占比下限（合并单元格 echo）
    _FORM_MIN_CHARS = 1000   # 短表格名单多为真实数据，交给 LLM 判断

    def _is_form_template(self, text: str) -> bool:
        if len(text) < self._FORM_MIN_CHARS:
            return False

        rows = [l for l in text.splitlines() if l.strip()]
        pipe_rows = [l for l in rows if l.count('|') >= 2]
        pipe_frac = sum(len(l) for l in pipe_rows) / max(len(text), 1)
        if pipe_frac < self._FORM_PIPE_MIN:
            return False

        narr = sum(m.end() - m.start() for m in _NARR_RE.finditer(text))
        if narr / max(len(text), 1) > self._FORM_NARR_MAX:
            return False

        cells = empty = dup = 0
        for l in pipe_rows:
            cs = [c.strip() for c in l.split('|')]
            if len(cs) >= 3 and not cs[0] and not cs[-1]:
                cs = cs[1:-1]
            cells += len(cs)
            empty += sum(1 for c in cs if not c)
            nonempty = Counter(c for c in cs if c)
            dup += sum(v for v in nonempty.values() if v >= 3)
        if cells == 0:
            return False
        return (empty / cells >= self._FORM_EMPTY_MIN
                or dup / cells >= self._FORM_DUP_MIN)

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