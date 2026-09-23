# cleaners/ — 清洗与启发式过滤层

读取层的文本先经过**确定性清洗**（无 API、可复现），再经过**启发式质量过滤**
（仍无 API）。两道都过关的文本才有资格进入付费的 LLM 质检（`quality_scorer.py`）。

```
pipeline._clean_all():
  HTML 来源     → HTMLCleaner.clean() → TextCleaner.clean()
  其余来源      → TextCleaner.clean()
pipeline._quality_check() 之前:
  QualityFilter.check() → 不过关 → rejected.jsonl(stage=clean)
```

## 文件说明

| 文件 | 职责 |
|------|------|
| `text_cleaner.py` | 通用文本清洗（9 步流水线，规则全部确定性） |
| `html_cleaner.py` | HTML 提取后文本的结构化收尾（残标签/表格线/标题空格/列表空行） |
| `quality_filter.py` | 文档级启发式过滤：长度、信息密度、乱码、语言、空表单熔断、SimHash 去重 |

## text_cleaner.py — 清洗 9 步（`clean()` 按序执行）

1. `_remove_control_chars`：删控制字符与**零宽字符**（U+200B-200F/U+2060/U+FEFF/软连字符）
   ——零宽符混在 wiki/OCR 文本里会让后面的"整行 UI 词"匹配失效，必须先删。
2. `_normalize_unicode`：**NFC 而非 NFKC**——NFKC 会把中文全角标点折叠成 ASCII
   （`，`→`,`），直接破坏中文语料标点。
3. `_normalize_whitespace`：全角空格→半角、连续空格并一、3+ 连续换行压到 2。
4. `_remove_ui_boilerplate`：删 wiki 界面残留（`[编辑]`、导航页脚行、固定短语表）；
   实测此类残留会真实拉低 LLM 的"语言规范性"评分，属清洗缺陷而非数据缺陷。
5. `_fix_line_breaks`：修复 PDF/网页硬断行——上一行不以句末标点结尾且两行都不是
   **结构化行**（`#` 标题 / `- *` 列表 / 编号行）时才合并；中文直接连写、英文补空格。
   结构化行豁免是 COT 样本数量的关键（实测修复后 COT 样本 6→12）。
6. `_remove_urls_and_emails`：URL→`[URL]`、邮箱→`[EMAIL]`（保留"这里有链接"的信号，不整体删除）。
7. `_remove_garbage_lines`：逐行统计非目标字符占比，>0.3 的行丢弃。
8. `_normalize_punctuation`：zh/mixed/unknown 保留中文标点只做 `...→…`；
   **仅纯英文文本**做全角→半角映射表转换。
9. `_collapse_repeated_chars`：同一字符连续 5+ 折叠为 2（`!!!!!!`→`!!`）。

## html_cleaner.py — 结构收尾

> ⚠️ 命名易误解：本模块**不解析 HTML、也不用 BeautifulSoup**
> （bs4 解析在 `readers/html_reader.py`）。它处理的是提取完的纯文本。
> 曾有一行从未使用的 `from bs4 import BeautifulSoup` 死导入，已清理。

- 残标签正则剥离（`<[^>]+>`、`[TABLE]` 标记）；
- 表格：删纯分隔线行 `|---|---|`，单元格压缩为 `A | B | C`；
- 标题：`##text` 修成 `## text`（转换器依赖 `^#{1,3} ` 匹配标题，缺空格会导致结构提取失败）；
- 列表项之间的多余空行压缩。

## quality_filter.py — 启发式过滤（`check()` 判定顺序即成本顺序）

```
empty → too_short(默认<30) → too_long(默认>1,000,000)
→ insufficient_information(前500字合法字符占比≤0.4)
→ garbage_content(三条规则,见下)
→ unknown_language(keep_unknown_language=False 时启用)
→ form_template_grid(空表单熔断)
→ duplicate_content(SimHash, 见 utils/dedup.py)
```
- **乱码三规则**（旧版"单字符出现>100 次即乱码"会把中文高频字误杀）：
  ① 连续重复 20+ 字符；② 最高频字符占比 >0.3（正常中文约 5%）；
  ③ 非允许字符集（ASCII+CJK 标点+表意区段白名单）占比 >0.3。
- **空表单熔断** `_is_form_template`：长度≥1000 且 管道表格行字符占比≥0.45
  且 叙述文本占比≤0.30（中文连续 8 字/英文连续三长词视为叙述）且
  （空单元格占比≥0.5 **或** 复读单元格占比≥0.4，后者针对合并单元格 echo）。
  在 400 文件真实语料标定：4 篇空白申报表全拦、248 篇已过检文档零误伤；
  填了真实数据的名单表（长实体名→叙述占比高、格子填满）天然不触发。
- `reset()`：清空去重历史（pipeline 在批次开始时调用）。

## 依赖

- 内部：`QualityFilter → utils.lang_detect`（语言判定）、`utils.dedup`（SimHash 去重）
- 外部：**无**（本层全部是确定性正则/统计规则；BeautifulSoup 依赖在 readers 层）
- 被 `pipeline.py` 引用；`rebuild_derived.py` 复用 TextCleaner/HTMLCleaner 从缓存重放清洗
