# readers/ — 文件读取层

把磁盘上的原始文件变成统一的 `ReaderResult`（纯文本 + 元数据 + 成败标记）。
**本层是唯一的 API 付费点之一**（Qwen-VL OCR），所以读取结果会被
`utils/cache.py` 落盘缓存，重跑不重复付费。

```
调用链：pipeline._read_single()
  → 按扩展名分发到 XxxReader.read()
  → ReaderResult(text, filepath, file_type, metadata, success, error)
```

## 文件说明

| 文件 | 职责 |
|------|------|
| `base.py` | `ReaderResult`（dataclass）与 `BaseReader` 抽象基类（`read` / `can_handle` / `get_file_type_label`） |
| `txt_reader.py` | `.txt`：chardet 探测编码（GB2312/GB18030→GBK 归一），`errors="replace"` 兜底读取 |
| `word_reader.py` | `.docx`：python-docx 提取段落 + 表格（表格行转 `A | B | C` 文本） |
| `pdf_reader.py` | `.pdf`：文字层优先，扫描版转 OCR（见下） |
| `html_reader.py` | `.html/.htm/.xhtml`：自动分流"源码 HTML / 截图类 HTML"两种提取策略 |
| `ocr_reader.py` | Qwen-VL 图片 OCR（DashScope `MultiModalConversation`），供截图 PNG、截图 HTML 内嵌图与扫描 PDF 复用 |

## 各读取器实现细节

### pdf_reader.py（分流逻辑最容易踩坑）
1. 先用 **pymupdf**（`import pymupdf as fitz`，兼容旧别名 `fitz`）逐页 `get_text()`；
   任何异常回退 **pypdf**。
2. 统计去掉空白后的字符数：`< ocr_min_text_chars`（默认 50）→ 判定为扫描版 PDF。
3. 扫描版走 `QwenVLOcrReader`：pymupdf 把每页渲染成 **300 DPI PNG**，
   写入 `ocr_temp/pages_<uuid>/`（刻意不用 `tempfile.mkdtemp`：部分沙箱拒绝
   mode 0700 目录），逐页 base64 送 Qwen-VL，`finally` 删临时目录。
4. `file_type` 区分来源方式：`pdf_download`（文字层）/ `pdf_photo`（OCR），
   元数据带 `page_count / used_ocr / char_count_before_ocr`。
5. **未配置 API key 时明确报错**（error 文案含 `DASHSCOPE_API_KEY`），
   不静默降级——扫描件"读不出"必须是显式失败而不是产出空文本。

### html_reader.py（两种 HTML 两套策略）
- `_is_screenshot_html()`：数特征投票——base64 内嵌图 >5、绝对定位、canvas、
  svg/path >20 记"截图分"；语义标签（article/section/h1-h6/p/li）记"源码分"，
  截图分 > 源码分且 ≥2 判为截图类。
- **源码 HTML**（`_extract_source_html`）：
  1. `decompose` 掉 script/style/nav/footer/header/aside/form/button 等噪声标签；
  2. 再剥离 MediaWiki 皮肤容器（`.mw-jump-link`、`#siteNotice`、`div.printfooter`、
     `#catlinks`、`.licensetpl`、`.mw-editsection`、`.vector-sticky-header` 等 20+ 选择器，
     逐条 try 防止非标准皮肤中断流程）；
  3. **按块级/行内区分缓冲**：行内节点（a/span/strong…）文本累积进当前段，
     块级节点（p/li/td/div…）才换段——旧实现把每个文本节点独立成段，
     `本文是<a>法律法规</a>。` 被切成三行，把结构/可读性评分全毁掉（真实教训）；
  4. 注释节点（NavigableString 子类，如 wikisource 尾部 parser-cache 注释）不是正文，排除。
- **截图类 HTML**（`_extract_screenshot_html`）：只取内嵌图片交给 OCR。

### ocr_reader.py
- 提示词固定为"**逐字提取**所有文字，保留段落/标题层级/表格/标点，不要总结不要遗漏"
  （不让模型做摘要，摘要会污染下游清洗与评分）。
- 每张图最多 `max_retries=3` 次，指数退避（1s/2s），最终失败返回空串（不抛异常，
  由 `read()` 转成失败结果）。
- Qwen-VL 偶尔返回 `[{text:…}, …]` 列表，`_extract_text` 统一拼回字符串。

## 依赖

- 内部：仅 `base.py` 的 `ReaderResult`；`ocr_reader` 被 `pdf_reader`/`html_reader` 组合使用
- 外部：`pymupdf|pypdf`、`python-docx`、`beautifulsoup4`、`chardet`、`dashscope`（仅 OCR 路径）
- 被 `pipeline.py` 引用；缓存写入在 pipeline 层做（reader 不感知缓存）
