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
        """将文本分块"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # langchain 要求 overlap < chunk_size，否则直接抛 ValueError；
        # 这里自动收敛，避免 --chunk-size 100 之类的配置导致整个 pipeline 崩溃
        chunk_size = max(1, int(self.chunk_size))
        overlap = max(0, min(int(self.chunk_overlap), chunk_size - 1))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", ".", "！", "？", " ", ""],
            length_function=len,
        )

        chunks = splitter.split_text(text)
        samples = []

        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk or len(chunk) < 10:
                continue

            chunk_lang = detect_language(chunk)
            samples.append({
                "text": chunk,
                "meta": {
                    "source": os.path.basename(result.filepath),
                    "type": result.file_type,
                    "language": chunk_lang,
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    "size_bytes": len(chunk.encode("utf-8")),
                    "token_count_est": estimate_token_count(chunk, chunk_lang),
                }
            })

        return samples