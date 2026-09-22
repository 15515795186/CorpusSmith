"""
语言检测工具
支持中文、英文混排检测
"""

import re
from typing import Optional


def detect_language(text: str) -> str:
    """检测文本主要语言: zh / en / mixed / unknown"""
    if not text or not text.strip():
        return "unknown"

    sample = text[:2000]
    total_chars = len(sample.strip())

    # 统计中文字符
    zh_chars = len(re.findall(r'[一-鿿]', sample))
    # 统计英文字母
    en_chars = len(re.findall(r'[a-zA-Z]', sample))
    # 统计数字
    digit_chars = len(re.findall(r'[0-9]', sample))

    meaningful = zh_chars + en_chars + digit_chars
    if meaningful == 0:
        return "unknown"

    zh_ratio = zh_chars / meaningful
    en_ratio = en_chars / meaningful

    if zh_ratio > 0.6:
        return "zh"
    elif en_ratio > 0.6:
        return "en"
    elif zh_ratio > 0.2 and en_ratio > 0.2:
        return "mixed"
    else:
        return "unknown"


def is_target_language(text: str, target_languages: list) -> bool:
    """检查文本是否属于目标语言"""
    lang = detect_language(text)
    if lang in target_languages:
        return True
    if lang == "mixed" and ("zh" in target_languages or "en" in target_languages):
        return True
    return False


def contains_chinese(text: str) -> bool:
    return bool(re.search(r'[一-鿿]', text))


def contains_english(text: str) -> bool:
    return bool(re.search(r'[a-zA-Z]{2,}', text))


def estimate_token_count(text: str, language: str = "auto") -> int:
    """
    估算 token 数量
    中文 ≈ 1.5 token/字, 英文 ≈ 1 token/4字符
    """
    if language == "auto":
        language = detect_language(text)

    if language == "zh":
        zh = len(re.findall(r'[一-鿿]', text))
        other = len(text) - zh
        return int(zh * 1.5 + other * 0.4)
    else:
        return int(len(text) * 0.35)