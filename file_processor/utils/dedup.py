"""
SimHash 去重工具
对文档级文本计算SimHash指纹，按阈值去重

设计（在真实语料上标定，见 README「去重标定」）：
- token 一律用 bigram：英文相邻词对 + 中文相邻字对。
  英文 unigram 会把不同论文误判为重复（同领域词表高度重叠，实测 12/50 误杀）；
  中文单字同理（实测 18/50 误杀）。
- 指纹必须用稳定内容哈希（blake2b），内置 hash() 带进程随机盐，结果不可复现。
- 高判重门槛 + 长度比约束：只清"近乎复制"的文档，保留同模板不同内容的文档
  （如不同大学的简介页）。
"""

import hashlib
import re
from typing import List, Set, Tuple


class SimHash:
    """SimHash 文本去重实现"""

    def __init__(self, f_size=64):
        self.f_size = f_size
        self._mask = (1 << f_size) - 1

    def _tokenize(self, text: str) -> List[str]:
        """相邻二元组：英文按词、中文按字"""
        tokens: List[str] = []
        en_words = re.findall(r'[a-zA-Z]{2,}', text.lower())
        tokens.extend(a + " " + b for a, b in zip(en_words, en_words[1:]))
        zh_chars = re.findall(r'[\u4e00-\u9fff]', text)
        tokens.extend(a + b for a, b in zip(zh_chars, zh_chars[1:]))
        return tokens

    def _hash(self, token: str) -> int:
        """稳定的 64 位哈希（不能用内置 hash()：跨进程随机盐导致不可复现）"""
        return int.from_bytes(
            hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big"
        ) & self._mask

    def fingerprint(self, text: str) -> int:
        """计算 SimHash 指纹"""
        v = [0] * self.f_size
        tokens = self._tokenize(text)
        if not tokens:
            return 0

        for token in tokens:
            h = self._hash(token)
            for i in range(self.f_size):
                bit = (h >> i) & 1
                if bit:
                    v[i] += 1
                else:
                    v[i] -= 1

        fingerprint = 0
        for i in range(self.f_size):
            if v[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint

    def hamming_distance(self, fp1: int, fp2: int) -> int:
        return bin(fp1 ^ fp2).count("1")

    def similarity(self, fp1: int, fp2: int) -> float:
        """计算两个指纹的相似度 (0~1)"""
        distance = self.hamming_distance(fp1, fp2)
        return 1.0 - distance / self.f_size


class Deduplicator:
    """文档级近重复去重器

    只判定"近乎复制"的文档：相似度 >= threshold（默认 0.90），
    且两者长度比 >= min_length_ratio（同模板、不同填充的文档长度往往
    差异明显，如不同大学的简介页，不应互判重复）。
    """

    MIN_TOKENS = 20  # token 太少时指纹无意义，不参与判重

    def __init__(self, threshold: float = 0.90, min_length_ratio: float = 0.6):
        self.threshold = threshold
        self.min_length_ratio = min_length_ratio
        self.simhash = SimHash()
        self.seen: Set[Tuple[int, int]] = set()  # (指纹, 文本长度)

    def is_duplicate(self, text: str) -> bool:
        """检查是否为重复文本；不重复时登记指纹"""
        if not text or len(text) < 50:
            return False

        fp = self.simhash.fingerprint(text[:5000])
        if fp == 0 or len(self.simhash._tokenize(text[:5000])) < self.MIN_TOKENS:
            return False

        for seen_fp, seen_len in self.seen:
            if min(len(text), seen_len) / max(len(text), seen_len) < self.min_length_ratio:
                continue
            if self.simhash.similarity(fp, seen_fp) >= self.threshold:
                return True

        self.seen.add((fp, len(text)))
        return False

    def reset(self):
        self.seen.clear()