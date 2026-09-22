"""
LLM 生成器基类
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMGenerator(ABC):
    """LLM生成器抽象接口"""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """同步生成"""
        ...

    @abstractmethod
    async def async_generate(self, prompt: str, system_prompt: str = "") -> str:
        """异步生成"""
        ...