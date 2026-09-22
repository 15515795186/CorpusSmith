"""
通用文本清洗器
处理所有类型文本的通用清洗规则
"""

import re
import unicodedata
from typing import Optional

from ..utils.lang_detect import detect_language

# 仅对英文文本生效：全角标点/字母/数字 → 半角
_FULLWIDTH_TO_ASCII = str.maketrans({
    "，": ",", "。": ".", "；": ";", "：": ":", "！": "!", "？": "?",
    "（": "(", "）": ")", "【": "[", "】": "]", "“": '"', "”": '"',
    "‘": "'", "’": "'",
    **{chr(0xFF10 + i): chr(0x30 + i) for i in range(10)},   # ０-９
    **{chr(0xFF21 + i): chr(0x41 + i) for i in range(26)},   # Ａ-Ｚ
    **{chr(0xFF41 + i): chr(0x61 + i) for i in range(26)},   # ａ-ｚ
})

# 结构化行：标题、列表项、编号步骤 —— 合并断行时必须保留换行，
# 否则 "1. xxx\n2. xxx" 会被粘成一行，COT/指令抽取将拿不到步骤结构。
_STRUCTURAL_LINE = re.compile(
    r'^\s*(?:#{1,6}\s|[-*•]\s|\d{1,3}[.、．)]\s?|[第][一二三四五六七八九十]+[步阶段环节]|'
    r'[（(]\d{1,3}[)）]|[一二三四五六七八九十]+[、.．]\s)')

# ---- wiki 界面残留（维基文库/维基百科下载件常见）----
# 整行只由界面词组成 → 删除（内容行含有这些词不受影响）
_UI_WORD = (r'(?:跳转到内容|跳至内容|跳转至导航|跳转到导航|跳转|导航|搜索|检索|目录|主页面?|首页|'
            r'特色内容|特色条目|随机条目|随机页面|贡献|创建账号|登录|注销|'
            r'用户页|用户讨论|讨论|话题|添加话题|阅读|查看来源|查看源代码|查看历史|编辑|'
            r'添加语言|语言|保存|取消|下载|打印|分享|外观|分类|姊妹计划|数据项|返回|'
            r'CentralNotice|siteNotice)')
_UI_LINE = re.compile(
    rf'^[\s|·／/、\-—]*(?:{_UI_WORD})(?:[\s|·／/、\-—]+(?:{_UI_WORD}))*[\s|·／/、\-—]*$')
# 整行是站点页脚/横幅/元数据
_UI_LINE_PATTERNS = re.compile(
    r'^\s*(?:'
    r'(?:从|来自)\s*[""「]?\S*(?:wikipedia|wikisource|wikimedia|wikibooks|wikiquote|wikiversity)\b.*$'
    r'|(?:从|来自)\s*[""「]?\S*(?:维基百科|维基文库|维基媒体|维基共享资源)\S*[""」]?\s*$'
    r'|(?:从|来自|維基).{0,8}(?:维基百科|维基文库|維基文庫|維基百科|Wikipedia|Wikisource)'
    r'[，,].{0,16}(?:自由|百科|图书|圖書館).{0,8}$'
    r'|^(?:维基百科|维基文库|維基文庫|維基百科|Wikipedia|Wikisource)[，,]?'
    r'(?:自由的?|the free).{0,20}$'
    r'|本页面最后修[订定于]+.{0,60}$'
    r'|本作品.{0,10}(?:公有领域|公有領域|CC\s*BY|知识共享|自由内容).{0,60}$'
    r'|(?:CentralNotice|siteNotice|Vectors\.search).{0,80}$'
    r'|^恭喜[！!].{0,24}(?:维基|維基|文库|文庫|文章|条目|達到|达到)'
    r'|^[\d〇零一二三四五六七八九十百千万亿两]{1,10}[篇条个页][！!]?$'
    r'|^(?:Public domain|PD-\w+|CC0|CC[ -]?BY(?:-SA)?|GNU Free Documentation'
    r'|Attribution-ShareAlike)(?:\s+\d+)?$'
    r'|^(?:true|false|none|None)$'
    r'|^https?://\S+$'
    r'|\[URL\]\s*$'
    r'|^Rendering was triggered.{0,60}$'
    r'|^(?:EPUB|MOBI|PDF|Ogg|DjVu|HTML|XML)$'
    r'|^用于(?:电脑|Kindle|大屏幕|打印|手机).{0,14}$'
    r'|^想要使用其他格式[？?]$'
    r'|^发生了一些错误$'
    r'|^[！!。．.？?，,、:：;；—\-•·]{1,3}$'
    r')', re.I)
# 行内 UI 标记：[编辑] [隐藏] [谁?] 等
_UI_INLINE = re.compile(
    r'\s*[\[（(]\s*(?:编辑|编辑源代码|隐藏|显示|关闭|谁\?|何时\?|来源请求|需要更好来源|'
    r'翻译|更新|原文|失效)\s*[\]）)]\s*')
# wikisource 皮肤固定文案（常被并进正文段落，行级规则失效，按字面清除）
_UI_PHRASES = (
    'EPUB 用于电脑、手机、平板', 'MOBI 用于Kindle', 'PDF 适用于大屏幕和打印',
    '想要使用其他格式？', '发生了一些错误返回', '发生了一些错误',
)
_UI_PHRASES_RE = re.compile(
    r'(?m)(?:本作品来自[^。\n]{0,40}。依据[^。\n]{0,80}属于公有领域。'
    r'|^Category[:：][^\n]*$'
    r'|Category[:：]\s?[^\s，。；、\n]{1,40}'
    r'|^下载(?=[\u4e00-\u9fff""][\u4e00-\u9fff""（）()0-9]{1,30}$))')


class TextCleaner:
    """通用文本清洗"""

    def __init__(self):
        pass

    def clean(self, text: str, language: Optional[str] = None) -> str:
        """完整清洗流水线

        language 未指定时自动检测：中文/混合文本保留中文标点，
        只有英文文本才做全角→半角转换（避免中文训练数据标点被破坏）。
        """
        if not text:
            return ""

        language = language or detect_language(text)

        text = self._remove_control_chars(text)
        text = self._normalize_unicode(text)
        text = self._normalize_whitespace(text)
        text = self._remove_ui_boilerplate(text)
        text = self._fix_line_breaks(text)
        text = self._remove_urls_and_emails(text)
        text = self._remove_garbage_lines(text)
        text = self._normalize_punctuation(text, language)
        text = self._collapse_repeated_chars(text)
        text = text.strip()

        return text

    def _remove_control_chars(self, text: str) -> str:
        """去除控制字符与零宽字符，保留 \\t \\n \\r

        零宽符（U+200B-200F/U+2060/U+FEFF）与软连字符常混在 wiki/OCR 文本里，
        会让"整行 UI 词"之类的行级匹配失效。
        """
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u2060\ufeff\u00ad]', '', text)

    def _normalize_unicode(self, text: str) -> str:
        """Unicode 标准化（NFC）

        注意：不能使用 NFKC —— 它会把中文全角标点（如 U+FF0C ，）折叠成
        ASCII 标点，破坏中文语料的标点。全角→半角只在英文文本上按需处理
        （见 _normalize_punctuation）。
        """
        return unicodedata.normalize('NFC', text)

    def _normalize_whitespace(self, text: str) -> str:
        """统一空白字符"""
        # 全角空格 → 半角
        text = text.replace('　', ' ')
        # 多个连续空格 → 单个
        text = re.sub(r'[ \t]+', ' ', text)
        # 多个连续空行 → 最多两个
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def _remove_ui_boilerplate(self, text: str) -> str:
        """删除 wiki 界面残留（下载件/HTML 转文本中的导航行、页脚、[编辑] 标记）

        质检实测此类残留会真实拉低语言规范性评分，属于清洗缺陷而非数据缺陷。
        """
        text = _UI_INLINE.sub(' ', text)
        for phrase in _UI_PHRASES:
            text = text.replace(phrase, ' ')
        text = _UI_PHRASES_RE.sub(' ', text)
        lines = []
        for line in text.split('\n'):
            stripped = line.strip()
            if not stripped:
                lines.append('')
                continue
            if _UI_LINE.match(stripped) or _UI_LINE_PATTERNS.match(stripped):
                continue
            lines.append(line)
        return '\n'.join(lines)

    def _fix_line_breaks(self, text: str) -> str:
        """修正断行问题（保留标题/列表/编号步骤等结构化行）"""
        # 英文单词断行修复: 单词断行 → 合并
        text = re.sub(r'(\w)-\n(\w)', r'\1\2\n\2', text)

        lines = text.split('\n')
        fixed = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                fixed.append('')
                continue

            prev = fixed[-1] if fixed else ''
            can_merge = (
                prev
                and not prev.endswith(('。', '！', '？', '.', '!', '?', '\n'))
                and not _STRUCTURAL_LINE.match(stripped)
                and not _STRUCTURAL_LINE.match(prev)
            )
            if can_merge:
                # 中文连写，英文补空格
                if re.match(r'[一-鿿]', stripped[-1:]) or re.match(r'[一-鿿]', prev[-1:]):
                    fixed[-1] = prev + stripped
                else:
                    fixed[-1] = prev + ' ' + stripped
            else:
                fixed.append(stripped)
        return '\n'.join(fixed)

    def _remove_urls_and_emails(self, text: str) -> str:
        """移除或替换URL和邮箱"""
        text = re.sub(r'https?://\S+', '[URL]', text)
        text = re.sub(r'[\w.-]+@[\w.-]+\.\w+', '[EMAIL]', text)
        return text

    def _remove_garbage_lines(self, text: str) -> str:
        """移除乱码行"""
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned.append('')
                continue
            # 统计非目标字符比例
            garbage = len(re.findall(r'[^\w\s一-鿿　-〿＀-￯'
                                     r'.,!?;:\'\"()\[\]{}\-–—=+*/@#$%^&<>~`|]', stripped))
            ratio = garbage / max(len(stripped), 1)
            if ratio > 0.3:
                continue  # 丢弃乱码行
            cleaned.append(stripped)
        return '\n'.join(cleaned)

    def _normalize_punctuation(self, text: str, language: Optional[str] = None) -> str:
        """标准化标点符号：中文/混合文本保留中文标点，英文文本转半角"""
        lang = (language or "").lower()
        if lang in ("zh", "mixed", "unknown", ""):
            return text.replace('...', '…')
        return text.translate(_FULLWIDTH_TO_ASCII)

    def _collapse_repeated_chars(self, text: str) -> str:
        """折叠过度重复的字符（如!!!!!! → !）"""
        text = re.sub(r'(.)\1{4,}', r'\1\1', text)
        return text


def clean_text(text: str, language: Optional[str] = None) -> str:
    """便捷函数"""
    cleaner = TextCleaner()
    return cleaner.clean(text, language)