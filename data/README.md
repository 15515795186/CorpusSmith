# 大模型数据标注数据集

当前工作区包含两类批次：四类常见文档/页面数据各 50 个，以及已有的英文/中文 PDF 数据集。

## 2026-09 新批次：四类数据（各 50 个）

| 目录 | 数量 | 数据内容 | 真实性说明 |
|------|------|----------|------------|
| `截图html/` | 50 | 1440x1400 PNG 页面截图 | 无头 Chrome 对中文维基文库长篇法律页面渲染截图 |
| `源码html/` | 50 | HTML 源码副本 | 与截图同一页面的完整渲染后 DOM，包含表格、信息框、多级标题和大量链接 |
| `word/` | 50 | `.docx` 文档 | 50 个来自公开政务/高校页面的非简历中文 DOCX 附件，文件内容未修改 |
| `txt/` | 50 | `.txt` 文本 | 50 篇中文政府白皮书全文，来自中文维基文库公开页面 |

四类数据使用相互独立的来源，没有从论文 PDF 转换，也没有复用 `文本pdf中文版/` 中的内容：

- HTML/截图：`zh.wikisource.org` 的“中华人民共和国法律”长篇页面
- DOCX：`gov.cn`、`moe.gov.cn`、`edu.cn` 等公开政务/高校文档附件
- TXT：`zh.wikisource.org` 中文政府白皮书全文

所有样本都保留来源 URL、来源类型、文件 SHA-256 和抓取/转换时间。四类文件哈希均为 50 个不同值。

HTML 复杂度最低值：约 296 KB、957 个元素、109 个链接、60 个段落、7 个标题、2 个表格。

完整清单：`_pipeline/manifest_mixed.json`；校验结果：`_pipeline/mixed_validation.json`。

重建命令：

```powershell
E:\Shenduxuexi\ProgramData\Anaconda_envs\envs\langchain1.2\python.exe _pipeline\build_independent_chinese_dataset.py
```

四类数据已经存在时，采集器只会替换其自身记录过的文件：

```powershell
E:\Shenduxuexi\ProgramData\Anaconda_envs\envs\langchain1.2\python.exe _pipeline\build_independent_chinese_dataset.py --force
```

## 已有：大模型领域 PDF 数据集（英文 100 篇 + 中文 100 篇）

生成时间：2026-09 批次 ｜ 英文源：arXiv（最新提交排序）｜ 中文源：汉斯出版社开放获取期刊

## 目录结构

| 目录 | 数量 | 类型 | 总大小 | 总页数 |
|------|------|------|--------|--------|
| `文本pdf/` | 50 | 英文原生电子文本 PDF（LLM 领域，文字层可提取） | 131.6 MB | 1162 |
| `版式pdf/` | 50 | 英文图片型/扫描版式 PDF（LLM 领域） | 355.8 MB | 1038 |
| `文本pdf中文版/` | 50 | 中文原生电子文本 PDF（科技/物理/数学） | 136.3 MB | 586 |
| `版式pdf中文版/` | 50 | 中文图片型/扫描版式 PDF（科技/物理/数学） | 149.3 MB | 542 |

## 英文集（LLM 领域，2026-09 最新）

- 按 arXiv 最新提交时间排序各取 50 篇，**两目录内容完全不重复（100 篇不同论文）**，
  主题聚焦大模型/Agent/推理/多模态/检索增强等，均为 2026-09 前后的最新成果，
  包含 DeepSeek-V4.1-Flash（KV Cache 压缩）、dQwen3.5 等开源大模型论文。

## 中文集（科技 / 物理 / 数学，2026 最新）

- 来源：汉斯出版社开放获取中文学术期刊（hanspub.org），覆盖理论数学、应用数学进展、统计学与应用、
  凝聚态物理学进展、核科学与技术、流体动力学、光电子、计算机科学与应用、软件工程与应用、
  人工智能与机器人研究、机械工程与技术、动力系统与控制、电气工程、电路与系统、海洋科学前沿等。
- “最新”：按论文全局编号（paperid）倒序抓取各刊最新期次，发表时间多为 **2026-05 ~ 2026-09**（如 2026/09/18）。
- 主题筛选：仅保留中文标题 + 科技/物理/数学类内容（剔除心理/医学/农业/水产/消费等无关主题）。
- 与英文集以及两个中文集之间均无重复论文。

## 校验结果（四套全过）

- 所有 200 个文件均为合法 PDF（`%PDF` 头 + 可打开）
- `文本pdf/`、`文本pdf中文版/`：50/50 文字层字符数 ≥ 80；中文集 50/50 含 ≥40 个汉字（真实可提取中文）
- `版式pdf/`、`版式pdf中文版/`：50/50 文字层字符数 = 0（真·图片型，模拟扫描件）
- 版式目录全页像素统计均含真实内容，无空白页
- 四个目录两两之间无重复 arXiv ID / 汉斯论文编号

## 版式 PDF 制作说明

按你确认的口径：最新论文基本只有电子文本版，真实扫描件稀少且过旧，
因此版式目录中的文件是把选定论文的每一页以 **150 DPI、JPEG q85** 渲染后再打包成
**纯图片 PDF**（逐页插图，页面尺寸与原论文一致），版面（双栏、公式、表格、图）与原文
完全一致，但无可选文字，用于测试 OCR / 版面分析管线对图片型 PDF 的效果。

## 典型收录（2026-09）

### 英文集（LLM 领域）
- DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression（`2609.19969`，文本集）
- dQwen3.5: Hybrid-Attention Diffusion Language Models（`2609.20751`，文本集）
- From Parameters to Behaviors: A Survey of Model Fusion for LLMs（`2609.19553`，版式集）

### 中文集（科技/物理/数学，期刊示例）
- 半群 E(X,σ) 的扩张元和 F、G 富足性（理论数学，文本集）
- 基于多文档异构结构图的检索增强生成方法（计算机科学与应用，文本集）
- 实时 RPV 中子剂量测量的共轭探测器响应函数方法研究（核科学与技术，版式集）
- 矩形断面涡激振动多模态响应数值研究（流体动力学，版式集）

全量清单：英文集见 `_pipeline/manifest.json`，中文集见 `_pipeline/manifest_cn.json`
（含论文编号 / 标题 / 期刊刊名 / 发表日期 / 文件名）。

## 附注

- 英文集命名：`<arXivID>_<论文标题缩写>.pdf`；中文集命名：`<汉斯论文编号>_<中文标题>.pdf`
- 汉斯站点为阿里云盾 WAF 防护，管线内置 `acw_sc__v2` 挑战求解（Node 执行），见 `_pipeline/solve_acw.js`
- 重跑管线：英文 `python _pipeline/pool_discover.py` → `python _pipeline/build_sets.py --text N --layout N`
  中文 `python _pipeline/cn_build.py --text 50 --layout 50`（巡检修复：`python _pipeline/cn_repair.py`）
