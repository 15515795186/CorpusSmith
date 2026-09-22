"""
通用文本清洗器
处理所有类型文本的通用清洗规则
"""

import re
import unicodedata
from typing import Optional

from ..utils.lang_detect import detect_language

# 仅对英文文本生效：全角标点/字母/数字 → 半角
_FULLWIDTH_TO_ASCII = str.maketrans({
    "，": ",", "。": ".", "；": ";", "：": ":", "！": "!", "？": "?",
    "（": "(", "）": ")", "【": "[", "】": "]", "“": '"', "”": '"',
    "‘": "'", "’": "'",
    **{chr(0xFF10 + i): chr(0x30 + i) for i in range(10)},   # ０-９
    **{chr(0xFF21 + i): chr(0x41 + i) for i in range(26)},   # Ａ-Ｚ
    **{chr(0xFF41 + i): chr(0x61 + i) for i in range(26)},   # ａ-ｚ
})

# 结构化行：标题、列表项、编号步骤 —— 合并断行时必须保留换行，
# 否则 "1. xxx\n2. xxx" 会被粘成一行，COT/指令抽取将拿不到步骤结构。
_STRUCTURAL_LINE = re.compile(
    r'^\s*(?:#{1,6}\s|[-*•]\s|\d{1,3}[.、．)]\s?|[第][一二三四五六七八九十]+[步阶段环节]|'
    r'[（(]\d{1,3}[)）]|[一二三四五六七八九十]+[、.．]\s)')


class TextCleaner:
    """通用文本清洗"""

    def __init__(self):
        pass

    def clean(self, text: str, language: Optional[str] = None) -> str:
        """完整清洗流水线

        language 未指定时自动检测：中文/混合文本保留中文标点，
        只有英文文本才做全角→半角转换（避免中文训练数据标点被破坏）。
        """
        if not text:
            return ""

        language = language or detect_language(text)

        text = self._remove_control_chars(text)
        text = self._normalize_unicode(text)
        text = self._normalize_whitespace(text)
        text = self._fix_line_breaks(text)
        text = self._remove_urls_and_emails(text)
        text = self._remove_garbage_lines(text)
        text = self._normalize_punctuation(text, language)
        text = self._collapse_repeated_chars(text)
        text = text.strip()

        return text

    def _remove_control_chars(self, text: str) -> str:
        """去除控制字符，保留 \\t \\n \\r"""
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    def _normalize_unicode(self, text: str) -> str:
        """Unicode 标准化（NFC）

        注意：不能使用 NFKC —— 它会把中文全角标点（如 U+FF0C ，）折叠成
        ASCII 标点，破坏中文语料的标点。全角→半角只在英文文本上按需处理
        （见 _normalize_punctuation）。
        """
        return unicodedata.normalize('NFC', text)

    def _normalize_whitespace(self, text: str) -> str:
        """统一空白字符"""
        # 全角空格 → 半角
        text = text.replace('　', ' ')
        # 多个连续空格 → 单个
        text = re.sub(r'[ \t]+', ' ', text)
        # 多个连续空行 → 最多两个
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def _fix_line_breaks(self, text: str) -> str:
        """修正断行问题（保留标题/列表/编号步骤等结构化行）"""
        # 英文单词断行修复: 单词断行 → 合并
        text = re.sub(r'(\w)-\n(\w)', r'\1\2\n\2', text)

        lines = text.split('\n')
        fixed = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                fixed.append('')
                continue

            prev = fixed[-1] if fixed else ''
            can_merge = (
                prev
                and not prev.endswith(('。', '！', '？', '.', '!', '?', '\n'))
                and not _STRUCTURAL_LINE.match(stripped)
                and not _STRUCTURAL_LINE.match(prev)
            )
            if can_merge:
                # 中文连写，英文补空格
                if re.match(r'[一-鿿]', stripped[-1:]) or re.match(r'[一-鿿]', prev[-1:]):
                    fixed[-1] = prev + stripped
                else:
                    fixed[-1] = prev + ' ' + stripped
            else:
                fixed.append(stripped)
        return '\n'.join(fixed)

    def _remove_urls_and_emails(self, text: str) -> str:
        """移除或替换URL和邮箱"""
        text = re.sub(r'https?://\S+', '[URL]', text)
        text = re.sub(r'[\w.-]+@[\w.-]+\.\w+', '[EMAIL]', text)
        return text

    def _remove_garbage_lines(self, text: str) -> str:
        """移除乱码行"""
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned.append('')
                continue
            # 统计非目标字符比例
            garbage = len(re.findall(r'[^\w\s一-鿿　-〿＀-￯'
                                     r'.,!?;:\'\"()\[\]{}\-–—=+*/@#$%^&<>~`|]', stripped))
            ratio = garbage / max(len(stripped), 1)
            if ratio > 0.3:
                continue  # 丢弃乱码行
            cleaned.append(stripped)
        return '\n'.join(cleaned)

    def _normalize_punctuation(self, text: str, language: Optional[str] = None) -> str:
        """标准化标点符号：中文/混合文本保留中文标点，英文文本转半角"""
        lang = (language or "").lower()
        if lang in ("zh", "mixed", "unknown", ""):
            return text.replace('...', '…')
        return text.translate(_FULLWIDTH_TO_ASCII)

    def _collapse_repeated_chars(self, text: str) -> str:
        """折叠过度重复的字符（如!!!!!! → !）"""
        text = re.sub(r'(.)\1{4,}', r'\1\1', text)
        return text


def clean_text(text: str, language: Optional[str] = None) -> str:
    """便捷函数"""
    cleaner = TextCleaner()
    return cleaner.clean(text, language)