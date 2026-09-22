"""
HTML 特定清洗器
对HTML提取后的文本做结构化解构
"""

import re
from bs4 import BeautifulSoup
from typing import Optional


class HTMLCleaner:
    """清洗HTML提取出的文本，还原为结构化Markdown风格"""

    def clean(self, html_text: str) -> str:
        if not html_text:
            return ""

        text = self._remove_html_tags(html_text)
        text = self._clean_table_markers(text)
        text = self._normalize_heading_levels(text)
        text = self._compact_list_items(text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _remove_html_tags(self, text: str) -> str:
        """移除残留的HTML标签"""
        # 处理 [TABLE] [/TABLE] 标记
        text = re.sub(r'\[/?TABLE\]', '', text)
        # 移除类似 <div>, </div> 标签
        text = re.sub(r'<[^>]+>', '', text)
        return text

    def _clean_table_markers(self, text: str) -> str:
        """清理表格分隔符"""
        # 移除纯分隔线行: |---|---|
        text = re.sub(r'\|[\s\-:]+\|[\s\-:]+\|', '', text)
        # 表格列合并
        lines = text.split('\n')
        cleaned = []
        in_table = False
        for line in lines:
            if '|' in line:
                if not in_table:
                    in_table = True
                # 清理表格行
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if cells:
                    cleaned.append(' | '.join(cells))
            else:
                if in_table:
                    in_table = False
                cleaned.append(line)
        return '\n'.join(cleaned)

    def _normalize_heading_levels(self, text: str) -> str:
        """标准化标题层级：修复 ##text → ## text 这类缺失空格的情况"""
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # 匹配 `#` 后紧跟非空白且非 `#` 的字符（即缺空格的情况）
            m = re.match(r'^(#{1,6})([^#\s])', stripped)
            if m:
                line = m.group(1) + ' ' + m.group(2)
            cleaned.append(line)
        return '\n'.join(cleaned)

    def _compact_list_items(self, text: str) -> str:
        """压缩列表项间距"""
        lines = text.split('\n')
        cleaned = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('- ') or stripped.startswith('* '):
                # 移除列表项间的空行
                if cleaned and cleaned[-1] == '':
                    if i > 0 and lines[i - 1].strip() == '':
                        cleaned.pop()
            cleaned.append(line)
        return '\n'.join(cleaned)


def clean_html_text(text: str) -> str:
    """便捷函数"""
    cleaner = HTMLCleaner()
    return cleaner.clean(text)