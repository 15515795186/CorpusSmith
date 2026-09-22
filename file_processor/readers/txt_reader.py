"""
纯文本文件读取器
自动检测编码 (UTF-8/GBK/GB2312)
"""

import os
import logging

from .base import BaseReader, ReaderResult

logger = logging.getLogger(__name__)


class TxtReader(BaseReader):
    """读取 .txt 纯文本文件"""

    def can_handle(self, filepath: str) -> bool:
        return filepath.lower().endswith('.txt')

    def _detect_encoding(self, filepath: str) -> str:
        """检测文件编码"""
        import chardet

        with open(filepath, "rb") as f:
            raw = f.read(min(10000, os.path.getsize(filepath)))

        result = chardet.detect(raw)
        encoding = result.get("encoding", "utf-8")

        # chardet 有时返回 None
        if encoding is None:
            encoding = "utf-8"

        # 标准化编码名称
        encoding = encoding.lower().replace("-", "")
        encoding_map = {
            "gb2312": "gbk",
            "gbk": "gbk",
            "gb18030": "gbk",
            "utf8": "utf-8",
            "ascii": "utf-8",
            "iso88591": "latin-1",
            "iso-8859-1": "latin-1",
        }
        return encoding_map.get(encoding, encoding)

    def read(self, filepath: str) -> ReaderResult:
        if not os.path.exists(filepath):
            return ReaderResult(
                text="", filepath=filepath, file_type="txt",
                success=False, error=f"File not found: {filepath}"
            )

        try:
            encoding = self._detect_encoding(filepath)

            with open(filepath, "r", encoding=encoding, errors="replace") as f:
                text = f.read()

            # 文件大小
            file_size = os.path.getsize(filepath)

            return ReaderResult(
                text=text,
                filepath=filepath,
                file_type="txt",
                metadata={
                    "encoding": encoding,
                    "file_size_bytes": file_size,
                },
                success=True,
            )
        except Exception as e:
            logger.exception("Failed to read TXT: %s", filepath)
            return ReaderResult(
                text="", filepath=filepath, file_type="txt",
                success=False, error=str(e)
            )