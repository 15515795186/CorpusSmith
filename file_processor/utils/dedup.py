"""
SimHash 去重工具
对文档级文本计算SimHash指纹，按阈值去重
"""

import re
from typing import List, Set


class SimHash:
    """SimHash 文本去重实现"""

    def __init__(self, f_size=64):
        self.f_size = f_size

    def _tokenize(self, text: str) -> List[str]:
        """分词，保留中英文"""
        # 中文按字切分，英文按词切分
        tokens = []
        # 提取英文单词
        en_words = re.findall(r'[a-zA-Z]{2,}', text.lower())
        tokens.extend(en_words)
        # 提取中文字符
        zh_chars = re.findall(r'[一-鿿]', text)
        tokens.extend(zh_chars)
        return tokens

    def _hash(self, token: str) -> int:
        """字符串 hash 为 f_size 位整数"""
        h = hash(token) & ((1 << self.f_size) - 1)
        return h

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
    """文档级去重器"""

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.simhash = SimHash()
        self.seen_fingerprints: Set[int] = set()

    def is_duplicate(self, text: str) -> bool:
        """检查是否为重复文本"""
        if not text or len(text) < 50:
            return False

        fp = self.simhash.fingerprint(text[:5000])
        for seen_fp in self.seen_fingerprints:
            sim = self.simhash.similarity(fp, seen_fp)
            if sim >= self.threshold:
                return True

        self.seen_fingerprints.add(fp)
        return False

    def reset(self):
        self.seen_fingerprints.clear()