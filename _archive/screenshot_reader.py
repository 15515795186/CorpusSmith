"""
截图/图片文件读取器（PNG截图）
使用 baidu/Unlimited-OCR 模型进行文字识别
"""

import os
import logging
from .base import BaseReader, ReaderResult
from ..utils.ocr_lock import ocr_load_lock

logger = logging.getLogger(__name__)


class ScreenshotReader(BaseReader):
    """读取 PNG 等截图/图片文件，通过 OCR 提取文字"""

    def __init__(self, ocr_model_name: str = "baidu/Unlimited-OCR",
                 ocr_device: str = "cuda",
                 ocr_cache_dir: str = r"D:\model"):
        self.ocr_model_name = ocr_model_name
        self.ocr_device = ocr_device
        self.ocr_cache_dir = ocr_cache_dir
        self._ocr_model = None
        self._ocr_tokenizer = None

    def can_handle(self, filepath: str) -> bool:
        return filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))

    def _load_ocr_model(self):
        if self._ocr_model is not None:
            return
        with ocr_load_lock:
            # Double-check after acquiring lock
            if self._ocr_model is not None:
                return
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                logger.info("Loading OCR model: %s", self.ocr_model_name)
                self._ocr_tokenizer = AutoTokenizer.from_pretrained(
                    self.ocr_model_name,
                    cache_dir=self.ocr_cache_dir,
                    trust_remote_code=True,
                )
                self._ocr_model = AutoModel.from_pretrained(
                    self.ocr_model_name,
                    cache_dir=self.ocr_cache_dir,
                    trust_remote_code=True,
                    use_safetensors=True,
                    dtype=torch.bfloat16,
                )
                if self.ocr_device == "cuda" and torch.cuda.is_available():
                    self._ocr_model = self._ocr_model.eval().cuda()
                    logger.info("OCR model loaded on GPU")
                else:
                    self._ocr_model = self._ocr_model.eval()
                    logger.info("OCR model loaded on CPU")
            except Exception as e:
                logger.error("Failed to load OCR model: %s", e)
                raise

    def read(self, filepath: str) -> ReaderResult:
        if not os.path.exists(filepath):
            return ReaderResult(
                text="", filepath=filepath, file_type="screenshot",
                success=False, error=f"File not found: {filepath}"
            )

        try:
            self._load_ocr_model()
            logger.info("OCR recognizing: %s", os.path.basename(filepath))

            result = self._ocr_model.infer(
                self._ocr_tokenizer,
                prompt="<image>document parsing.",
                image_file=filepath,
                output_path="./ocr_temp",
                base_size=1024,
                image_size=640,
                crop_mode=True,
                max_length=32768,
                no_repeat_ngram_size=35,
                ngram_window=128,
            )

            text = str(result) if result else ""
            file_size = os.path.getsize(filepath)

            return ReaderResult(
                text=text,
                filepath=filepath,
                file_type="screenshot",
                metadata={
                    "file_size_bytes": file_size,
                    "ocr_model": self.ocr_model_name,
                },
                success=True,
            )
        except Exception as e:
            logger.warning("OCR failed for screenshot: %s - %s",
                           os.path.basename(filepath), e)
            return ReaderResult(
                text="", filepath=filepath, file_type="screenshot",
                success=False, error=str(e)
            )