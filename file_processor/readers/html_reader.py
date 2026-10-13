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
from bs4.element import Comment, CData, Doctype, ProcessingInstruction

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

    # lxml 在部分页面上会整棵丢掉正文: 实测 gov.cn 源码 html
    # BeautifulSoup(raw,'lxml').get_text()=188 字符 vs 'html.parser'=4517 字符,
    # 正文全在 <p> 里却在 lxml 树中不存在 —— 8/10 份政务 html 语料因此零产出
    # (reader 只吐出 "…_中国政府网EN个人中心退出邮箱无障碍EN" 这样的导航碎片)。
    _MIN_ALT_RATIO = 0.02      # 提取文本 < 源文 2% → 视为解析异常
    _BRANCH_MIN_CHARS = 300    # 通道提取量低于此值 → 怀疑通道误判

    def _parse_html(self, html_content: str, label: str = "") -> BeautifulSoup:
        """解析 HTML, 并在 lxml 明显丢正文时回退到 html.parser"""
        soup = BeautifulSoup(html_content, 'lxml')
        if len(html_content) < 2000:
            return soup
        got = len(soup.get_text().strip())
        if got >= max(200, len(html_content) * self._MIN_ALT_RATIO):
            return soup
        alt = BeautifulSoup(html_content, 'html.parser')
        alt_len = len(alt.get_text().strip())
        if alt_len > got * 2:
            logger.warning("lxml 解析丢正文(%d 字符 / 源 %d), 回退 html.parser(%d 字符): %s",
                           got, len(html_content), alt_len, label or "?")
            return alt
        return soup

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

    _HEADINGS = ('h1', 'h2', 'h3', 'h4', 'h5', 'h6')
    _BLOCK_START = {'p', 'div', 'li', 'td', 'th', 'section', 'article', 'main',
                    'blockquote', 'dd', 'dt', 'pre', 'figcaption', 'center'}

    def _extract_source_html(self, soup: BeautifulSoup) -> str:
        """从源码HTML中提取语义化文本

        关键设计：按 块级/行内 区分——行内节点（a/span/strong…）的文本
        累积进当前段落缓冲，块级节点（p/li/td/div…）开始新段落。
        旧实现对每个文本节点直接 "\n\n".join，导致含链接的句子被切碎成
        "本文是 / 中国内地法律法规 / 。" 三段，结构与可读性全毁。
        """
        # 移除噪声元素
        for tag in soup.find_all(['script', 'style', 'noscript', 'iframe',
                                  'svg', 'nav', 'footer', 'header',
                                  'aside', 'form', 'button', 'select']):
            tag.decompose()

        # MediaWiki 皮肤容器（wikisource Vector 皮肤源码实测定位）：
        # 跳转链接、站点通知/横幅、许可证模板元数据、下载工具项、
        # 分类框、页脚、编辑小节、侧栏与吸顶栏等
        for sel in ('.mw-jump-link', '#mw-aria-live-region', '#siteSub',
                    '.mw-dismissable-notice', '#siteNotice', '#localNotice',
                    '.anonnotice', 'div.printfooter', '#catlinks', '.catlinks',
                    '.licensetpl', '.licensetpl_header', 'li.noprint',
                    'li.mw-list-item', '.mw-editsection', '#p-lang',
                    '.vector-user-links', '.vector-sticky-header',
                    'div.vector-header-container', 'table.metadata',
                    '.mw-authority-control', '#section-description'):
            try:
                for tag in soup.select(sel):
                    tag.decompose()
            except Exception:
                pass  # 非标准皮肤下选择器失效不影响主流程

        # <head> 整体移除（title 已由 read() 先行提取），
        # 注释/CDATA 等 NavigableString 子类不是正文
        # （wikisource 页面尾部的 parser-cache 注释含 "Rendering was triggered" 噪声）
        for head in soup.find_all('head'):
            head.decompose()
        for node in soup.find_all(string=lambda s: isinstance(s, (Comment, CData, Doctype, ProcessingInstruction))):
            node.extract()

        text_parts: list = []
        buf: list = []
        current_block = None

        def flush():
            if not buf:
                return
            para = re.sub(r'\s+', ' ', ' '.join(buf)).strip()
            # CJK 相邻片段之间的连接空格去掉（中英混排保留）
            para = re.sub(r'(?<=[\u4e00-\u9fff。，、；：？！）》」』】])'
                          r'\s+(?=[\u4e00-\u9fff（《「『【])', '', para)
            # 中文标点回贴：'法律法规 。' → '法律法规。'
            para = re.sub(r'\s+([，。；：、！？）》」』%])', r'\1', para)
            para = re.sub(r'([（《「『])\s+', r'\1', para)
            if para and para.strip('-•·—'):
                text_parts.append(para)
            buf.clear()

        for element in soup.descendants:
            if not isinstance(element, NavigableString):
                continue
            text = element.strip()
            if not text:
                continue
            parent = element.parent
            tag = parent.name if parent is not None else None

            if tag in self._HEADINGS:
                flush()
                current_block = None
                text_parts.append("#" * int(tag[1]) + " " + text)
            elif tag == 'li':
                if current_block is not parent:
                    flush()
                    current_block = parent
                    buf.append('-')
                buf.append(text)
            elif tag is None or tag in self._BLOCK_START or tag in ('br',):
                if current_block is not parent:
                    flush()
                    current_block = parent
                buf.append(text)
            else:
                # 行内元素（a/span/strong/em/b/i/u/small…）及未知标签：
                # 并入当前段落，不产生段落边界
                buf.append(text)
        flush()

        full_text = "\n\n".join(text_parts)
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

            soup = self._parse_html(html_content, os.path.basename(filepath))
            is_screenshot = self._is_screenshot_html(soup)

            # 标题先提取：_extract_source_html 会移除整个 <head>
            title_tag = soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else ""

            if is_screenshot:
                text = self._extract_screenshot_html(soup)
                file_type = "html_screenshot"
                logger.info("Detected screenshot HTML: %s", os.path.basename(filepath))
            else:
                text = self._extract_source_html(soup)
                file_type = "html_source"
                logger.info("Detected source HTML: %s", os.path.basename(filepath))

            # 通道误判回退: 源码型政务页面带 position:absolute 内联样式, 会被判成截图 HTML,
            # 而截图通道只拿到导航碎片; 两条通道都试, 取正文更多者(实测 8/10 份政务页).
            if len(html_content) >= 2000 and len(text.strip()) < self._BRANCH_MIN_CHARS:
                alt_soup = self._parse_html(html_content, os.path.basename(filepath))
                alt_text = (self._extract_source_html(alt_soup) if is_screenshot
                            else self._extract_screenshot_html(alt_soup))
                if len(alt_text.strip()) > len(text.strip()) * 2:
                    logger.warning("HTML 通道被判为 %s 但仅取到 %d 字符, 改用 %s 通道(%d 字符): %s",
                                   file_type, len(text.strip()),
                                   "html_source" if is_screenshot else "html_screenshot",
                                   len(alt_text.strip()), os.path.basename(filepath))
                    text, file_type = alt_text, ("html_source" if is_screenshot else "html_screenshot")
                    is_screenshot = not is_screenshot

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