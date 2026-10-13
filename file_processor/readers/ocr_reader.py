"""
基于通义千问 Qwen-VL 的 OCR 识别读取器
用于 截图PNG 和 扫描PDF 渲染出的页面图片
"""

import base64
import logging
import mimetypes
import os
import time

from .base import BaseReader, ReaderResult

logger = logging.getLogger(__name__)

DEFAULT_OCR_PROMPT = (
    "请逐字提取图片中的所有文字，保留原始段落结构、标题层级、表格格式和标点符号。"
    "不要总结，不要遗漏。"
)

_MIME_FALLBACK = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}


class QwenVLOcrReader(BaseReader):
    """使用通义千问 Qwen-VL 做OCR识别的通用读取器"""

    def __init__(self, api_key: str = "", model: str = "qwen-vl-plus",
                 prompt: str = DEFAULT_OCR_PROMPT,
                 max_retries: int = 3, max_tokens: int = 8000):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.model = model
        self.prompt = prompt
        self.max_retries = max_retries
        # 长图(整页截图)必须显式抬输出上限: 不传时用服务端默认(约 1-2k tokens),
        # 整页文字会在句中被腰斩 → 下游质量评委判"真实截断"整份淘汰
        self.max_tokens = max_tokens

    def can_handle(self, filepath: str) -> bool:
        return filepath.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))

    # ---------- 图片编码 ----------
    @staticmethod
    def _mime_type(image_path: str) -> str:
        ext = os.path.splitext(image_path)[1].lower()
        return _MIME_FALLBACK.get(ext) or mimetypes.guess_type(image_path)[0] or "image/png"

    def _image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # ---------- OCR ----------
    def ocr_image(self, image_path: str) -> str:
        """对单张图片OCR，返回文本（失败返回空字符串）"""
        if not self.api_key:
            raise RuntimeError(
                "未配置 DashScope API key，无法调用 Qwen-VL OCR "
                "(请设置环境变量 DASHSCOPE_API_KEY)")

        import dashscope
        from dashscope import MultiModalConversation

        dashscope.api_key = self.api_key
        data_uri = f"data:{self._mime_type(image_path)};base64," \
                   f"{self._image_to_base64(image_path)}"
        messages = [{
            "role": "user",
            "content": [
                {"image": data_uri},
                {"text": self.prompt},
            ]
        }]

        last_error = ""
        for attempt in range(self.max_retries):
            try:
                response = MultiModalConversation.call(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                )
                if response.status_code == 200:
                    return self._extract_text(response.output.choices[0].message.content)
                last_error = f"{getattr(response, 'code', '')} - {response.message}"
                logger.warning("DashScope API attempt %d/%d failed: %s",
                               attempt + 1, self.max_retries, last_error)
            except Exception as e:
                last_error = str(e)
                logger.warning("DashScope API attempt %d/%d error: %s",
                               attempt + 1, self.max_retries, e)

            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)

        logger.error("Qwen-VL OCR 放弃 (%s): %s", os.path.basename(image_path), last_error)
        return ""

    @staticmethod
    def _extract_text(content) -> str:
        """Qwen-VL 有时返回 list，提取其中的文本片段"""
        if isinstance(content, list):
            texts = [item['text'] for item in content
                     if isinstance(item, dict) and 'text' in item]
            return '\n'.join(texts)
        return content or ""

    # ---------- 读取 ----------
    def read(self, filepath: str) -> ReaderResult:
        if not os.path.exists(filepath):
            return ReaderResult(
                text="", filepath=filepath, file_type="screenshot",
                success=False, error=f"File not found: {filepath}"
            )

        try:
            logger.info("Qwen-VL OCR: %s", os.path.basename(filepath))
            text = self.ocr_image(filepath)
            if not text.strip():
                return ReaderResult(
                    text="", filepath=filepath, file_type="screenshot",
                    success=False, error="OCR 返回空文本",
                    metadata={"ocr_model": self.model},
                )

            return ReaderResult(
                text=text,
                filepath=filepath,
                file_type="screenshot",
                metadata={
                    "file_size_bytes": os.path.getsize(filepath),
                    "ocr_model": self.model,
                },
                success=True,
            )
        except Exception as e:
            logger.warning("Qwen-VL OCR failed: %s - %s",
                           os.path.basename(filepath), e)
            return ReaderResult(
                text="", filepath=filepath, file_type="screenshot",
                success=False, error=str(e)
            )