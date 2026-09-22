"""
OpenAI 兼容 API 生成器
支持 OpenAI、以及任何兼容的 API（如 DeepSeek、通义千问兼容模式）
"""

import os
import time
import logging
from typing import Optional

from .base import BaseLLMGenerator

logger = logging.getLogger(__name__)


class OpenAIGenerator(BaseLLMGenerator):
    """OpenAI 兼容 API 调用"""

    def __init__(self,
                 api_key: str = "",
                 base_url: str = "",
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.3,
                 max_tokens: int = 2048,
                 max_retries: int = 3):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """同步生成"""
        from openai import OpenAI

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client = OpenAI(**client_kwargs)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning("OpenAI API attempt %d/%d failed: %s",
                               attempt + 1, self.max_retries, e)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    async def async_generate(self, prompt: str, system_prompt: str = "") -> str:
        """异步生成"""
        from openai import AsyncOpenAI

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client = AsyncOpenAI(**client_kwargs)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning("OpenAI async API attempt %d/%d failed: %s",
                               attempt + 1, self.max_retries, e)
                if attempt < self.max_retries - 1:
                    await self._async_sleep(2 ** attempt)
                else:
                    raise

    async def _async_sleep(self, seconds: float):
        import asyncio
        await asyncio.sleep(seconds)