"""
预训练数据集转换器
将清洗后的文本转换为预训练格式（纯文本 + 元数据）
"""

import os
import re
import logging
from typing import List, Dict, Optional

from ..utils.lang_detect import detect_language, estimate_token_count
from ..readers.base import ReaderResult

logger = logging.getLogger(__name__)


class PretrainConverter:
    """转换为预训练数据集格式"""

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128,
                 max_samples: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_samples = max_samples

    def convert(self, result: ReaderResult) -> List[Dict]:
        """将读取结果转换为预训练样本列表"""
        if not result.success or not result.text:
            return []

        text = result.text.strip()
        language = detect_language(text)
        token_count = estimate_token_count(text, language)

        if self.chunk_size <= 0:
            # 整文档输出
            samples = [{
                "text": text,
                "meta": {
                    "id": f"{os.path.basename(result.filepath)}#0",
                    "source": os.path.basename(result.filepath),
                    "type": result.file_type,
                    "language": language,
                    "size_bytes": len(text.encode("utf-8")),
                    "token_count_est": token_count,
                }
            }]
        else:
            # 分块输出
            samples = self._chunk_text(text, result, language, token_count)

        # 限制样本数
        if self.max_samples > 0:
            samples = samples[:self.max_samples]

        return samples

    def _chunk_text(self, text: str, result: ReaderResult,
                     language: str, total_tokens: int) -> List[Dict]:
        """将文本分块（标准库实现，替代 langchain_text_splitters 依赖）"""
        chunk_size = max(1, int(self.chunk_size))
        overlap = max(0, min(int(self.chunk_overlap), chunk_size - 1))

        separators = ["\n\n", "\n", "。", "！", "？", ".", " ", ""]
        pieces = self._plain_split(text, chunk_size, separators)
        chunks = self._greedy_merge(pieces, chunk_size, overlap)

        samples = []
        base = os.path.basename(result.filepath)
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk or len(chunk) < 10:
                continue

            chunk_lang = detect_language(chunk)
            samples.append({
                "text": chunk,
                "meta": {
                    "id": f"{base}#{i}",
                    "source": base,
                    "type": result.file_type,
                    "language": chunk_lang,
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    "size_bytes": len(chunk.encode("utf-8")),
                    "token_count_est": estimate_token_count(chunk, chunk_lang),
                }
            })

        return samples

    @staticmethod
    def _plain_split(text: str, chunk_size: int,
                     separators: List[str]) -> List[str]:
        """递归展开：超长片段逐级按 separators 切分（与 langchain 的
        RecursiveCharacterTextSplitter 同思路），直到全部片段 ≤ chunk_size。"""
        pieces = [text]
        for sep in separators:
            if sep == "" or all(len(p) <= chunk_size for p in pieces):
                break
            exp = []
            for p in pieces:
                if len(p) <= chunk_size:
                    exp.append(p)
                else:
                    exp.extend(x for x in re.split(re.escape(sep), p) if x)
            pieces = exp
        # 兜底：单片段仍超长（如超长无分隔词）按字符硬切
        hard = []
        for p in pieces:
            if len(p) > chunk_size:
                hard.extend(p[i:i + chunk_size] for i in range(0, len(p), chunk_size))
            else:
                hard.append(p)
        return hard

    @staticmethod
    def _greedy_merge(pieces: List[str], chunk_size: int,
                      overlap: int) -> List[str]:
        """贪心合并片段为 ≤ chunk_size 的块；相邻块重叠 overlap 字符
        （与 langchain 语义近似；overlap 守卫已在 _chunk_text 收敛到 < chunk_size）。"""
        chunks, cur, cur_len = [], [], 0
        for p in pieces:
            if len(p) >= chunk_size:  # 硬切片段自成一块
                if cur:
                    chunks.append("".join(cur))
                    cur, cur_len = [], 0
                chunks.append(p)
                continue
            if cur_len + len(p) > chunk_size and cur:
                block = "".join(cur)
                chunks.append(block)
                tail = block[-overlap:] if overlap > 0 and len(block) >= overlap else ""
                cur = [tail] if tail else []
                cur_len = len(tail)
            cur.append(p)
            cur_len += len(p)
        if cur:
            chunks.append("".join(cur))
        return [c for c in chunks if c.strip()]