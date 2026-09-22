"""
DashScope (通义千问) API 生成器
"""

import os
import json
import logging
from typing import Optional

from .base import BaseLLMGenerator

logger = logging.getLogger(__name__)


class DashScopeGenerator(BaseLLMGenerator):
    """阿里云 DashScope（通义千问）调用"""

    def __init__(self,
                 api_key: str = "",
                 model: str = "qwen-max",
                 temperature: float = 0.3,
                 max_tokens: int = 2048,
                 max_retries: int = 3):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """同步生成"""
        import dashscope
        from dashscope import Generation

        dashscope.api_key = self.api_key
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = Generation.call(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    result_format="message",
                )
                if response.status_code == 200:
                    return response.output.choices[0].message.content or ""
                else:
                    raise Exception(f"DashScope API error: {response.code} - {response.message}")
            except Exception as e:
                logger.warning("DashScope API attempt %d/%d failed: %s",
                               attempt + 1, self.max_retries, e)
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
                else:
                    raise

    async def async_generate(self, prompt: str, system_prompt: str = "") -> str:
        """异步生成（DashScope也支持异步）"""
        import dashscope
        from dashscope import Generation

        import asyncio
        dashscope.api_key = self.api_key
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = Generation.call(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    result_format="message",
                )
                if response.status_code == 200:
                    return response.output.choices[0].message.content or ""
                else:
                    raise Exception(f"DashScope API error: {response.code} - {response.message}")
            except Exception as e:
                logger.warning("DashScope async API attempt %d/%d failed: %s",
                               attempt + 1, self.max_retries, e)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise