"""
HTML 文件读取器
- HTML源码: 提取语义文本（保留标题/列表/表格结构）
- 截图HTML: 提取可见文本层，过滤定位/样式噪声
自动区分两类HTML
"""

import os
import re
import logging
from bs4 import BeautifulSoup, NavigableString

from .base import BaseReader, ReaderResult

logger = logging.getLogger(__name__)


class HTMLReader(BaseReader):
    """
    HTML 读取器
    - 自动检测截图HTML vs 源码HTML
    - 对截图HTML做更强力的噪声过滤
    """

    def can_handle(self, filepath: str) -> bool:
        return filepath.lower().endswith(('.html', '.htm', '.xhtml'))

    def _is_screenshot_html(self, soup: BeautifulSoup) -> bool:
        """判断是否为截图类HTML（通过特征检测）"""
        html_str = str(soup)

        # 截图HTML特征: 大量内联base64图片、绝对定位、canvas
        features = [
            len(re.findall(r'data:image/\w+;base64,', html_str)) > 5,
            bool(re.search(r'position\s*:\s*absolute', html_str)),
            bool(re.findall(r'<canvas', html_str, re.I)),
            bool(re.findall(r'style\s*=\s*"[^"]*left\s*:', html_str)),
            bool(re.findall(r'style\s*=\s*"[^"]*top\s*:', html_str)),
            len(soup.find_all(['svg', 'path'])) > 20,
        ]

        # 源码HTML特征: 语义化标签
        semantic_features = [
            bool(soup.find_all(['article', 'section', 'nav', 'header', 'footer', 'main'])),
            bool(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
            len(soup.find_all('p')) > 3,
            bool(soup.find_all(['ul', 'ol', 'li'])),
        ]

        score_screenshot = sum(features)
        score_source = sum(semantic_features)

        return score_screenshot > score_source and score_screenshot >= 2

    def _extract_source_html(self, soup: BeautifulSoup) -> str:
        """从源码HTML中提取语义化文本"""
        # 移除噪声元素
        for tag in soup.find_all(['script', 'style', 'noscript', 'iframe',
                                  'svg', 'nav', 'footer', 'header',
                                  'aside', 'form', 'button', 'select']):
            tag.decompose()

        # 提取文本块
        text_parts = []

        for element in soup.descendants:
            if isinstance(element, NavigableString):
                parent = element.parent
                if parent and parent.name:
                    text = element.strip()
                    if not text:
                        continue
                    tag = parent.name

                    if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        level = tag[1]
                        text_parts.append("#" * int(level) + " " + text)
                    elif tag == 'li':
                        text_parts.append("- " + text)
                    elif tag in ['td', 'th']:
                        text_parts.append(text)
                    elif tag == 'p':
                        text_parts.append(text)
                    elif tag == 'br':
                        text_parts.append("\n")
                    elif tag in ['div', 'span', 'a', 'strong', 'em', 'b', 'i', 'u']:
                        text_parts.append(text)

        full_text = "\n\n".join(text_parts)

        # 清理多余空行
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        return full_text.strip()

    def _extract_screenshot_html(self, soup: BeautifulSoup) -> str:
        """从截图HTML中提取可见文本"""
        # 移除所有样式、脚本
        for tag in soup.find_all(['script', 'style', 'noscript', 'svg',
                                  'meta', 'link', 'base', 'canvas']):
            tag.decompose()

        # 移除所有class和id属性（保留文本）
        for tag in soup.find_all(True):
            attrs_to_remove = ['class', 'id', 'style', 'onclick', 'onload',
                               'onmouseover', 'onmouseout', 'data-*']
            for attr in list(tag.attrs.keys()):
                if attr.startswith('on') or attr in ['class', 'id', 'style',
                                                      'data-src', 'data-id']:
                    del tag[attr]

        # 获取所有文本
        text = soup.get_text(separator='\n')

        # 清理：移除坐标类文本、单字符行、纯数字行
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # 过滤：纯数字/坐标行
            if re.match(r'^[\d\s,.\-px%]+$', stripped) and len(stripped) < 50:
                continue
            # 过滤：路径类
            if re.match(r'^[A-Z]?:\\\\', stripped):
                continue
            # 过滤：过短的行（可能是噪声）
            if len(stripped) < 2:
                continue
            cleaned_lines.append(stripped)

        return '\n'.join(cleaned_lines)

    def read(self, filepath: str) -> ReaderResult:
        if not os.path.exists(filepath):
            return ReaderResult(
                text="", filepath=filepath, file_type="html",
                success=False, error=f"File not found: {filepath}"
            )

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, 'lxml')
            is_screenshot = self._is_screenshot_html(soup)

            if is_screenshot:
                text = self._extract_screenshot_html(soup)
                file_type = "html_screenshot"
                logger.info("Detected screenshot HTML: %s", os.path.basename(filepath))
            else:
                text = self._extract_source_html(soup)
                file_type = "html_source"
                logger.info("Detected source HTML: %s", os.path.basename(filepath))

            # 提取标题
            title_tag = soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else ""

            return ReaderResult(
                text=text,
                filepath=filepath,
                file_type=file_type,
                metadata={
                    "title": title,
                    "is_screenshot": is_screenshot,
                    "original_length": len(html_content),
                },
                success=True,
            )
        except Exception as e:
            logger.exception("Failed to read HTML: %s", filepath)
            return ReaderResult(
                text="", filepath=filepath, file_type="html",
                success=False, error=str(e)
            )