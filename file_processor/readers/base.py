"""
读取器基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ReaderResult:
    """读取器的统一输出"""
    text: str                                      # 提取的纯文本
    filepath: str                                  # 源文件路径
    file_type: str                                 # 文件类型分类
    metadata: dict = field(default_factory=dict)   # 元数据（页数、语言等）
    images: List[str] = field(default_factory=list) # OCR用到的图片路径列表
    success: bool = True
    error: Optional[str] = None


class BaseReader(ABC):
    """读取器基类"""

    @abstractmethod
    def read(self, filepath: str) -> ReaderResult:
        """读取文件，返回提取结果"""
        ...

    def get_file_type_label(self, filepath: str) -> str:
        """根据文件路径推断所属大类分类"""
        path_lower = filepath.lower()
        if path_lower.endswith('.pdf'):
            return "pdf"
        elif path_lower.endswith('.docx'):
            return "word"
        elif path_lower.endswith(('.html', '.htm', '.xhtml')):
            return "html"
        elif path_lower.endswith('.txt'):
            return "txt"
        else:
            return "unknown"

    def can_handle(self, filepath: str) -> bool:
        """检查能否处理此文件"""
        raise NotImplementedError