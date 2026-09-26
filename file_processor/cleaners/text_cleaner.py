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
            # 政务门户页导航(实测: reader 输出头部 "首页 | 简 | 繁 | EN | 登录个人中心退出 | 邮箱 | 无障碍 EN")
            r'简|繁|EN|邮箱|无障碍|个人中心|退出|收藏|打印|'
            r'CentralNotice|siteNotice)')
# 分隔符用 * 而非 +: 政务门户导航常是连写的("- | 登录个人中心退出"), 实测该行
# 因词间无分隔而被漏删; 代价是"纯界面词连写行"也会被删(本就是页眉页脚)
_UI_LINE = re.compile(
    rf'^[\s|·／/、\-—]*(?:{_UI_WORD})(?:[\s|·／/、\-—]*(?:{_UI_WORD}))*[\s|·／/、\-—]*$')
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
    # 政务门户页眉/按钮文案(实测 reader 输出: "字号：默认大超大 | 打印 收藏 | 索引号：")
    # 注意: '收藏'/'打印' 是正常词汇(实测误删过 "国家美术发展和收藏工程"), 只能走行级规则
    '字号：默认大超大', '字号：默认大更大', '大超大',
)
_UI_PHRASES_RE = re.compile(
    r'(?m)(?:本作品来自[^。\n]{0,40}。依据[^。\n]{0,80}属于公有领域。'
    r'|^Category[:：][^\n]*$'
    r'|Category[:：]\s?[^\s，。；、\n]{1,40}'
    r'|^下载(?=[\u4e00-\u9fff""][\u4e00-\u9fff""（）()0-9]{1,30}$))')
# 政务门户面包屑("首页 > 政策 > 国务院政策文件库 > 国务院文件"): 只剥"段 > 段"前缀。
# 历史缺陷: 初版正则连带吞掉了正文("首页&nbsp;&gt;&nbsp;政策文件" 被吃成 "文件"),
# 故改为逐段匹配(每段 ≤12 个汉字, 以 '>' 收尾), 剩余为空或纯界面词时才整行删。
_BREADCRUMB_RE = re.compile(
    r'^[>\s·]*(?:[\u4e00-\u9fffA-Za-z0-9]{2,16}\s*>\s*){1,8}[^\n]*$')

# ---- 提取噪声修复（PDF 文字层 / 截图 OCR / 网页稿实测）----
# 命运多舛的一批规则，全部有实测样本支撑（见 tests/test_cleaners.py 的 ExtractionNoiseTest）：
#   1) 连字残留: PDF 文字层给的是 ﬁ ﬂ ﬀ（U+FB00-FB06），不展开会污染英文语料
#   2) 坏变音符: 变音符号被拆成独立字符（"J\u00a8urgen" = Jürgen），
#      但 "ˆξ"/"˜a" 是 LaTeX 数学帽子/波浪号，**不能动** → 只修 ¨ ´ 后接元音
#   3) HTML 实体: 网页稿里 "&nbsp;&gt;&nbsp;" 是导航面包屑，必须先解码再做行级规则
#   4) 截图 OCR 会整篇返回带坐标的 HTML 转储（"<div class=\"line\" data-bbox=\"58 31 ...\">"）
_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
              "\ufb04": "ffl", "\ufb05": "ft", "\ufb06": "st"}
# 仅 ¨(分音符) / ´(重音符) + 元音 —— 高置信的"坏掉的名字/借词"
_DIACRITIC_FIXES = {
    "\u00a8a": "ä", "\u00a8o": "ö", "\u00a8u": "ü",
    "\u00a8A": "Ä", "\u00a8O": "Ö", "\u00a8U": "Ü",
    "\u00b4e": "é", "\u00b4a": "á", "\u00b4o": "ó", "\u00b4u": "ú",
    "\u00b4E": "É", "\u00b4A": "Á", "\u00b4O": "Ó", "\u00b4U": "Ú",
}
_HTML_ENTITIES = {"&nbsp;": " ", "&ensp;": " ", "&emsp;": " ", "&gt;": ">",
                  "&lt;": "<", "&amp;": "&", "&quot;": '"', "&#39;": "'", "&apos;": "'"}
# 截图 OCR 的 HTML 转储特征（正文论文里的 "<API>"/"<ATP>" 不含这些特征，不会被误处理）
_MARKUP_DUMP_RE = re.compile(r'data-bbox|<html\s*>\s*<body|^```html', re.M)
# 网页稿（浏览器另存/抓取件）的元数据头与站点导航残留
_SNAPSHOT_HEAD_RE = re.compile(
    r'^\s*(?:来源|标题|网址|URL|链接|发布时间|成文日期|发文机关|索引号|主题分类)\s*[:：]')
_SNAPSHOT_CHROME_RE = re.compile(
    r'[_＿]?(?:国务院文件|国务院部门文件)?[_＿]?中国政府网(?:EN个人中心退出邮箱无障碍EN)?'
    r'|EN?个人中心退出邮箱无障碍EN?')

# 断词合并的**保留白名单**(只列几乎不可能是长词起始片段的组合前缀)。
#
# 标定过程(全库 962 处跨行连字符审计): 初版把虚词(in/be/an/re/per/the/of/for…)也当"应保留",
# 结果 115 处"保留"里约 95 处是**假保留**(in-formation / be-tween / an-swer / re-spectively —— 
# 这些虚词恰好是长词的起始片段), 假保留比假合并多 4 倍。故策略改为**合并偏向**:
# 只保留 self-/non-/multi-/cross- 这类几乎不会被长词撞车的组合前缀, 其余一律合并
# (tak-ing → taking, lan-guage → language, Tool-former → Toolformer)。
# 代价: "state-of-the-art" 会成为 "stateof-the-art"、"Common-sense" → "Commonsense", 量极少。
_HYPHEN_KEEP = frozenset(('self', 'non', 'multi', 'cross'))


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
        text = self._decode_html_entities(text)          # 实体解码须在行级规则之前, 否则 &nbsp; 干扰行级判定
        text = self._unwrap_markup_dump(text)            # 截图 OCR 的 HTML 转储 → 纯文本
        text = self._fix_ligatures_and_diacritics(text)  # 连字/坏变音符(须在断行修复前)
        text = self._strip_snapshot_chrome(text)         # 网页稿元数据头/站点导航残留
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
        # 私有区字形(U+E000-F8FF)是图标字体的产物(截图 OCR 实测: 收藏/打印等图标变成 \ue68a),
        # 不是可训练文本
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u2060\ufeff\u00ad\ue000-\uf8ff]', '', text)

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
            # 政务门户面包屑("首页 > 政策 > 国务院政策文件库 > 国务院文件"): 整行/行首片段
            # 整行呈面包屑形状("首页 > 政策 > 国务院政策文件库 > …")才删:
            # 段长 ≥2 字符, 故 '见 > 第 3 节' 这类正文不会被误删
            if _BREADCRUMB_RE.match(stripped):
                continue
            lines.append(line)
        return '\n'.join(lines)

    def _fix_line_breaks(self, text: str) -> str:
        """修正断行问题（保留标题/列表/编号步骤等结构化行）"""
        # 英文行尾连字符断词: "tak-\ning a random" → "taking a random"。
        # 历史缺陷: 旧式 r'(\w)-\n(\w)' → r'\1\2\n\2' 只取换行后的**一个字母**复制,
        # 于是 "tak-\ning" 变成 "taki" + 残行 "ng", 行合并再补空格 → 全库制造出
        # "taki ing"/"dist tribution"/"sett tings" 这类**假断词**(实测 4299 处跨行拼接候选),
        # 质检把它当成 OCR 错误点名扣分。正确做法: 吃掉 "-\n" 让词完整接回下一行。
        # 仅当下行以小写字母开头才考虑合并; 且若下行词首是英文虚词/短词
        # ("state-\nof-the-art"、"self-\nattention" 的 of/self 类), 判定为**真连字符**,
        # 保留连字符 —— 否则会把 "state-of-the-art" 粘成 "stateof-the-art"。
        def _dehyph(m):
            prev_word, frag = m.group(1), m.group(2)
            head = re.match(r"[a-z]+", frag).group(0)
            # 前一个词是前缀/虚词(self-\nattention / state-\nof-the-art)或后段词首是虚词
            # → 真连字符, 保留连字符; 否则视为 PDF 软连字符, 合并(tak-\ning → taking)。
            keep = prev_word in _HYPHEN_KEEP or head in _HYPHEN_KEEP
            return prev_word + ('-' if keep else '') + frag
        text = re.sub(r'([A-Za-z]+)-\n[ \t]*([a-z][a-z\-]*)', _dehyph, text)

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
                if prev.endswith('-') and re.match(r'[A-Za-z0-9]', stripped[:1]):
                    # 真连字符跨行("COVID-\n19"/"state-\nof-the-art"): 保留连字符且不补空格
                    fixed[-1] = prev + stripped
                # 中文连写，英文补空格
                elif re.match(r'[一-鿿]', stripped[-1:]) or re.match(r'[一-鿿]', prev[-1:]):
                    fixed[-1] = prev + stripped
                else:
                    fixed[-1] = prev + ' ' + stripped
            else:
                fixed.append(stripped)
        return '\n'.join(fixed)

    # ---- 提取噪声修复 ----

    def _decode_html_entities(self, text: str) -> str:
        """HTML 实体解码 + 不换行空格归一

        实测(政务网页稿 txt): "&nbsp;&gt;&nbsp;" 组成的面包屑导航与标题行里
        夹着 10~134 个实体, 不解码则行级/短语级残留规则全部失效, 质检直接点名
        "混入大量 HTML/网页渲染残留"。缺了这一步, 后面的清洗规则形同虚设。
        """
        if '&' not in text:
            return text
        for ent, ch in _HTML_ENTITIES.items():
            text = text.replace(ent, ch)
        text = re.sub(r'&#(\d{2,6});',
                      lambda m: chr(int(m.group(1))) if int(m.group(1)) < 0x110000 else m.group(0), text)
        text = re.sub(r'&#[xX]([0-9a-fA-F]{2,6});',
                      lambda m: chr(int(m.group(1), 16)) if int(m.group(1), 16) < 0x110000 else m.group(0), text)
        return text.replace('\xa0', ' ')

    def _unwrap_markup_dump(self, text: str) -> str:
        """截图 OCR 整篇返回带坐标的 HTML 转储 → 还原为纯文本

        实测(截图 OCR): 返回物是 python-markdown 代码块包裹的 HTML:
        ```html\n<html><body>\n<div class="line" data-bbox="58 31 307 52">中华人民共和国中央人民政府</div> …
        直接入库等于把 255 处标签属性当正文(质检: "大量HTML标签和data-bbox属性残留")。
        只取标签之间的文本。注意: 论文正文里的 "<API>"/"<ATP>" 是合法内容,
        它们不含 data-bbox/<html><body> 特征, 不会被本规则碰到(有反例测试守着)。
        """
        if not _MARKUP_DUMP_RE.search(text):
            return text
        text = re.sub(r'```[a-zA-Z]*', ' ', text)
        inner = re.findall(r'>([^<>]+)<', text)
        if len(inner) >= 3:
            kept = [x.strip() for x in inner if x.strip()]
            if kept:
                text = '\n'.join(kept)
        return re.sub(r'<[^<>]{0,200}>', ' ', text)

    def _fix_ligatures_and_diacritics(self, text: str) -> str:
        """连字展开 + 坏变音符修复(只修高置信的两类)

        实测(英文 arXiv 文字层 PDF): 连字 ﬁ ﬂ ﬀ 残留 62~284 处/份;
        变音符号被拆成独立字符 117 处/份。但绝大多数 "ˆX"/"˜a" 是 LaTeX 数学
        帽子与波浪号(如 ˆξ、˜a), **不能映射成字母** —— 只修 ¨/´ + 元音
        ("J\u00a8urgen" → Jürgen, "J\u00b4egou" → Jégou)。
        """
        if any(c in text for c in _LIGATURES):
            text = text.translate(str.maketrans(_LIGATURES))
        for broken, fixed in _DIACRITIC_FIXES.items():
            if broken in text:
                text = text.replace(broken, fixed)
        return text

    def _strip_snapshot_chrome(self, text: str) -> str:
        """网页稿元数据头 + 站点导航残留

        实测(政务网页稿 txt): 头部是抓取元数据("来源: <url>" / "标题: <文件名>"),
        正文里夹着 "_中国政府网EN个人中心退出邮箱无障碍EN" 这类站点导航与面包屑,
        质检点名 "网页爬虫残留内容"。标题行尾部还挂着 "_国务院部门文件_中国政府网"。
        """
        lines = text.split('\n')
        kept, head_budget = [], 6      # 只在文件头 6 行内剥元数据, 避免误删正文里的 "标题：" 讨论
        for line in lines:
            stripped = line.strip()
            if head_budget > 0 and stripped and _SNAPSHOT_HEAD_RE.match(stripped):
                head_budget -= 1
                continue
            if stripped:
                head_budget -= 1 if head_budget > 0 else 0
            kept.append(line)
        text = '\n'.join(kept)
        text = _SNAPSHOT_CHROME_RE.sub(' ', text)
        # 只剩标点/尖括号碎片的行(网页稿里常见的 \"/> 残片)
        text = re.sub(r'(?m)^[\s\'"/>]*[\'"/>][\s\'"/>]*$', '', text)
        return text

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