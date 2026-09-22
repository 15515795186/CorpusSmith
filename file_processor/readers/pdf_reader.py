"""
PDF 文件读取器
- 文字PDF: 使用 pymupdf（回退 pypdf）直接提取
- 扫描/图片PDF: 渲染页面后通过 Qwen-VL API 逐页 OCR

说明：本地 baidu/Unlimited-OCR 方案已废弃（见 _archive/），
本项目统一走 DashScope Qwen-VL；未配置 API key 时扫描版 PDF 会明确报错，
而不是静默尝试加载本地模型。
"""

import logging
import os
import shutil
import uuid
from typing import Optional

from .base import BaseReader, ReaderResult

logger = logging.getLogger(__name__)


def _import_fitz():
    """兼容 PyMuPDF 1.24+ 的新包名 pymupdf 与旧别名 fitz"""
    try:
        import pymupdf as fitz
    except ImportError:  # pragma: no cover - 取决于安装版本
        import fitz
    return fitz


class PDFReader(BaseReader):
    """支持文字PDF和扫描PDF的读取器"""

    def __init__(self, min_text_chars: int = 50,
                 qwen_ocr: Optional[object] = None,
                 ocr_dpi: int = 300,
                 work_dir: str = "ocr_temp"):
        self.min_text_chars = min_text_chars
        self.qwen_ocr = qwen_ocr  # QwenVLOcrReader 实例
        self.ocr_dpi = ocr_dpi
        self.work_dir = work_dir  # 渲染页面的临时目录（工作区内，便于清理）

    def can_handle(self, filepath: str) -> bool:
        return filepath.lower().endswith('.pdf')

    # ---------- 文本提取 ----------
    def _extract_text_pymupdf(self, filepath: str) -> str:
        fitz = _import_fitz()
        doc = fitz.open(filepath)
        try:
            return "\n\n".join(page.get_text() for page in doc)
        finally:
            doc.close()

    def _extract_text_pypdf(self, filepath: str) -> str:
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)

    def _extract_text(self, filepath: str) -> str:
        try:
            return self._extract_text_pymupdf(filepath)
        except Exception as e:  # pymupdf 不可用或文件异常 → 回退 pypdf
            logger.warning("pymupdf 提取失败(%s)，回退 pypdf: %s",
                           os.path.basename(filepath), e)
            return self._extract_text_pypdf(filepath)

    def page_count(self, filepath: str) -> int:
        fitz = _import_fitz()
        doc = fitz.open(filepath)
        try:
            return len(doc)
        finally:
            doc.close()

    # ---------- OCR ----------
    def _ocr_pages_qwen(self, filepath: str) -> str:
        """用 Qwen-VL API 逐页 OCR（页面渲染为临时 PNG）"""
        fitz = _import_fitz()
        doc = fitz.open(filepath)
        total = len(doc)
        text_parts = []
        # 不用 tempfile.mkdtemp：部分沙箱会拒绝写入 mode 0700 的目录
        temp_dir = os.path.join(self.work_dir, f"pages_{uuid.uuid4().hex[:10]}")
        os.makedirs(temp_dir, exist_ok=True)
        try:
            for page_num in range(total):
                pix = doc[page_num].get_pixmap(dpi=self.ocr_dpi)
                img_path = os.path.join(temp_dir, f"page_{page_num:04d}.png")
                pix.save(img_path)
                logger.info("Qwen-VL OCR page %d/%d (%s)",
                            page_num + 1, total, os.path.basename(filepath))
                text_parts.append(self.qwen_ocr.ocr_image(img_path))
        finally:
            doc.close()
            shutil.rmtree(temp_dir, ignore_errors=True)
        return "\n\n".join(text_parts)

    # ---------- 主流程 ----------
    def read(self, filepath: str) -> ReaderResult:
        if not os.path.exists(filepath):
            return ReaderResult(
                text="", filepath=filepath, file_type="pdf",
                success=False, error=f"File not found: {filepath}"
            )

        try:
            raw_text = self._extract_text(filepath)
            char_count = len(raw_text.strip().replace("\n", "").replace(" ", ""))
            used_ocr = False

            if char_count < self.min_text_chars:
                logger.info("Detected scanned PDF (chars=%d < %d), running OCR...",
                            char_count, self.min_text_chars)
                if self.qwen_ocr is None:
                    return ReaderResult(
                        text="", filepath=filepath, file_type="pdf_photo",
                        success=False,
                        error="扫描版PDF需要OCR，但未配置 DashScope API key "
                              "(请设置 DASHSCOPE_API_KEY)",
                        metadata={"char_count_before_ocr": char_count},
                    )
                try:
                    raw_text = self._ocr_pages_qwen(filepath)
                    used_ocr = True
                except Exception as ocr_err:
                    logger.warning("OCR failed for %s: %s",
                                   os.path.basename(filepath), ocr_err)
                    return ReaderResult(
                        text="", filepath=filepath, file_type="pdf_photo",
                        success=False, error=f"OCR failed: {ocr_err}",
                        metadata={"char_count_before_ocr": char_count},
                    )

            return ReaderResult(
                text=raw_text,
                filepath=filepath,
                file_type="pdf_photo" if used_ocr else "pdf_download",
                metadata={
                    "page_count": self.page_count(filepath),
                    "used_ocr": used_ocr,
                    "char_count_before_ocr": char_count,
                },
                success=True,
            )
        except Exception as e:
            logger.exception("Failed to read PDF: %s", filepath)
            return ReaderResult(
                text="", filepath=filepath, file_type="pdf",
                success=False, error=str(e)
            )