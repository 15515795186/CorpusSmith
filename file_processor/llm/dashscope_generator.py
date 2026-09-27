"""
DashScope (通义千问) API 生成器
通过 OpenAI 兼容端点 (https://dashscope.aliyuncs.com/compatible-mode/v1) 调用。
同一端点可调用百炼托管的 DeepSeek 系列模型（deepseek-r1 / deepseek-v3 等）。
"""

import os
import time
import logging

from .base import BaseLLMGenerator

logger = logging.getLogger(__name__)

COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

#: 各 provider 的 OpenAI 兼容端点
PROVIDER_BASE_URLS = {
    "dashscope": COMPATIBLE_BASE_URL,
    "deepseek": "https://api.deepseek.com/",
}
#: 各 provider 的密钥环境变量名
PROVIDER_ENV_KEYS = {
    "dashscope": "DASHSCOPE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


class DashScopeGenerator(BaseLLMGenerator):
    """阿里云 DashScope（通义千问/百炼）调用，OpenAI 兼容模式

    model 可为: qwen3.5-plus / qwen3.5-max / qwen-plus / deepseek-r1 / deepseek-v3 等
    （以百炼控制台实际开通的模型为准）
    """

    def __init__(self,
                 api_key: str = "",
                 model: str = "qwen3.5-plus",
                 temperature: float = 0.3,
                 max_tokens: int = 2048,
                 max_retries: int = 3,
                 provider: str = "dashscope",
                 base_url: str = ""):
        self.provider = provider if provider in PROVIDER_BASE_URLS else "dashscope"
        env_key = PROVIDER_ENV_KEYS.get(self.provider, "DASHSCOPE_API_KEY")
        self.api_key = api_key or os.getenv(env_key, "")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        # 端点优先级: 显式 base_url > provider 专属环境变量 > 内置表
        env_base = os.getenv(self.provider.upper() + "_BASE_URL", "")
        self.base_url = base_url or env_base or PROVIDER_BASE_URLS[self.provider]

    def _request_kwargs(self, messages) -> dict:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        # DeepSeek 推理模型（官方/百炼托管）不支持 temperature，传入会报 400
        if not self.model.lower().startswith("deepseek"):
            kwargs["temperature"] = self.temperature
        return kwargs

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """同步生成"""
        from openai import OpenAI

        # SDK 层: 180s 超时 + 关闭 SDK 内置重试(外层 max_retries 循环已管重试;
        # 60s 对 deepseek-r1 4000-token 长链生成太紧, 会把能成功的请求掐死;
        # max_retries=0 防 http.client 复用挂死等待放大)
        client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                        timeout=180, max_retries=0)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    **self._request_kwargs(messages)
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning("%s API attempt %d/%d failed: %s", self.provider,
                               attempt + 1, self.max_retries, e)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    async def async_generate(self, prompt: str, system_prompt: str = "") -> str:
        """异步生成"""
        import asyncio
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url,
                             timeout=180, max_retries=0)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = await client.chat.completions.create(
                    **self._request_kwargs(messages)
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning("%s async API attempt %d/%d failed: %s", self.provider,
                               attempt + 1, self.max_retries, e)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise


#: 语义化别名 —— 本类实为 "OpenAI 兼容端点" 通用实现(DeepSeek / DashScope 通用)
OpenAICompatGenerator = DashScopeGenerator