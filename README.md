# LLM 数据集处理工作流

批量读取 400 份多格式文档（PDF / Word / HTML / TXT / PNG），经 **解析 / OCR → 清洗去重 → LLM 质量评分**，
转换为 **3 种大模型训练数据集**：预训练（pretrain）、指令微调（SFT）、Chain-of-Thought（COT）。

---

## 核心特性

| 特性 | 说明 |
|------|------|
| 增量可复用 | 读取结果（含 OCR）落盘缓存，重跑不再重复解析、不重复调用付费 API |
| 断点续跑 | `--resume` 在**读取之前**就跳过已产出与确定性淘汰的来源；质检因限流淘汰的来源会自动重试 |
| 结果可复现 | 文件按路径排序处理；每次运行重建输出，不再出现跨次运行记录堆叠 |
| 可诊断 | 被丢弃文本写入 `rejected.jsonl` 并统计原因，便于定位阈值/OCR 质量问题 |
| 确定性产出 | 每次运行生成 `run_report.json`（文件数、缓存命中、丢弃原因、数据集规模、脱敏配置） |
| 凭据安全 | 项目根 `.env` 自动加载（gitignore 覆盖），真实环境变量优先；key 永不写入代码/仓库/注册表 |
| 无外部依赖测试 | 116 个单元 + 端到端测试（标准库 unittest），不需要 API key |

---

## 目录结构

```
LLM_Data_Test/
├── file_processor/                  # 核心处理包（32 个 py / 4268 行，结构见图后导航）
│   ├── README.md                    # ★ 包内总览：数据流、依赖表、根文件逐个说明
│   ├── main.py                      # CLI 入口
│   ├── pipeline.py                  # 管道编排：扫描→读取→清洗→质检→转换→输出
│   ├── config.py                    # 全局配置（dataclass + JSON 覆盖）
│   ├── dotenv.py                    # 零依赖 .env 加载器（真实环境变量优先）
│   ├── quality_scorer.py            # deepseek-flash 多维度质量评分（并发）
│   ├── run_config_v3.json           # 运行配置：全流程（最终语料 v3 的 24 份 → 三份产出）
│   ├── run_config_cot_sft.json      # 运行配置：只跑 COT（以既有 SFT 问答对为素材）
│   ├── readers/                     # pdf / word / html / txt / qwen-vl ocr（各有 README）
│   ├── cleaners/                    # text_cleaner / html_cleaner / quality_filter（README）
│   ├── converters/                  # pretrain / instruction / cot（README：含 input 契约）
│   ├── formatters/jsonl_formatter.py# 输出布局 + 重建/续跑/元数据（README）
│   ├── llm/                         # 可选：dashscope 生成器（README）
│   └── utils/                       # lang_detect / dedup / stats / cache（README）
├── data/                            # 原始数据 + 数据构建工具链（详见 data/README.md）
│   ├── 爬取语料_v3/                 ★ 项目最终输入：8 类 × 3 = 24 份（见下方说明）
│   ├── 爬取语料_v2/                 第二版（40 份，备用/测试）
│   ├── 爬取语料/                    初版（80 份，测试用例）
│   ├── 文本pdf/                     50 篇英文论文 PDF（文字层，初版测试语料）
│   ├── 文本pdf中文版/                50 篇中文论文 PDF（文字层）
│   ├── 版式pdf/                     50 篇英文扫描版 PDF（需 OCR）
│   ├── 版式pdf中文版/                50 篇中文扫描版 PDF（需 OCR）
│   ├── word/                       50 份政务/高校 DOCX
│   ├── 源码html/                    50 个维基法律页面 HTML
│   ├── 截图html/                    50 张页面截图 PNG（需 OCR）
│   ├── txt/                        50 篇政府白皮书 TXT
│   ├── tech_output/                数据构建暂存区
│   ├── 爬取语料/                    初版爬取语料（80 份）
│   ├── 爬取语料_v2/                 第二版（40 份，备用）
│   ├── 爬取语料_v3/                 第三版（8 类 × 3 = 24 份，当前主用；清单见 data/_pipeline/manifest_v3.json）
│   └── _pipeline/                  采集/生成/校验脚本（重建数据时使用）
├── datasets_v3_20260927/            # 当前产出（pretrain 237 / instruction 165 / cot 51；目录名由 output_dir 配置）
├── docs/                            # 分析留档（对账_COT第三轮_20260927.md、薄答案实例_20260927.md）
├── tests/                           # 单元与端到端测试
├── _archive/                        # 已废弃的本地 OCR 方案（见其 README）
├── check_progress.py                # 进度查看：输入量 / 产出 / 缓存 / 报告 / 日志（自动选最新 ocr_run*.log）
├── verify_datasets.py               # 产出自检：重复记录 / 字段缺失 / 编码 / 分块 / metadata 口径
├── rebuild_derived.py               # 转换器改动后从读取缓存免费重建 instruction/cot（不动 pretrain/质检结论，零 API）
├── run_quality_score.py             # 对已有数据集重新评分（独立口径，不破坏原文件）
├── .env.example                     # 凭据模板（复制为 .env 填 key；.env 已被 gitignore）
├── requirements.txt
└── README.md
```

### 模块实现细节导航

想看清"某一步具体怎么实现"，按层进对应 README（含判定顺序、阈值标定依据、踩坑记录）：

| 文档 | 回答的问题 |
|------|-----------|
| [`file_processor/README.md`](file_processor/README.md) | 一次运行经过哪些代码？谁 import 谁？成本闸门怎么排？ |
| [`readers/README.md`](file_processor/readers/README.md) | 扫描 PDF 怎么判定并 OCR？HTML 两种形态怎么分流？ |
| [`cleaners/README.md`](file_processor/cleaners/README.md) | 9 步清洗每步防什么？空表单熔断规则怎么标定到零误伤？ |
| [`converters/README.md`](file_processor/converters/README.md) | 三种样本各怎么提取？**input 什么时候必须给材料**？各种守卫拦掉了什么噪声？ |
| [`formatters/README.md`](file_processor/formatters/README.md) | 重建/续跑语义、样本级指纹去重、metadata 累计口径 |
| [`llm/README.md`](file_processor/llm/README.md) | 可选生成路径；为什么质检/OCR 不走它 |
| [`utils/README.md`](file_processor/utils/README.md) | SimHash 为何必须 bigram+blake2b？缓存键为何含 mtime？ |

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API key（三选一，不要写进代码）
#    a. 推荐：项目根放 .env（已被 gitignore），运行时自动加载
#       PowerShell: Copy-Item .env.example .env  然后填入 key
#       需要两个: DEEPSEEK_API_KEY(文本链: SFT/COT 生成、质量评分、语义抽检)
#                 DASHSCOPE_API_KEY(仅 OCR 视觉链 qwen-vl-plus; 可留空, 走读取缓存即可)
#    b. 临时：当前窗口设置，进程退出即消失
export DEEPSEEK_API_KEY="sk-xxx"           # Linux/macOS
$env:DEEPSEEK_API_KEY="sk-xxx"             # Windows PowerShell
#    c. 不推荐：写入用户环境变量（明文长期存注册表，所有子进程可见）
setx DEEPSEEK_API_KEY "sk-xxx"             # 优先级：真实环境变量 > .env 文件

# 3. 先看配置与待处理文件（不写任何文件）
python -m file_processor.main --dry-run

# 4. 直接跑生产（默认参数就是 v3 最终配置：最终语料 24 份 → pretrain / instruction / cot）
python -m file_processor.main

# 5. 只跑 COT（以既有 SFT 问答对为素材: 判类 → 回原文搜据 → 写规范字段）
python -m file_processor.main --config file_processor/run_config_cot_sft.json

# 6. 清空重建（默认 resume=true 只补不覆盖, 与最终配置一致）
python -m file_processor.main --rebuild

# 7. 查看进度
# 8. 查看进度
python check_progress.py
```

**入口与输出目录**

- **不带任何参数 = 生产运行**：输入 = 项目最终语料 `data/爬取语料_v3`（8 类 × 3 = 24 份），
  输出 = `datasets_v3_20260927/`，`resume=true`（只补不覆盖；要清空重建加 `--rebuild`）。
- `ProcessorConfig()` 的默认值**就是 v3 最终配置**（逐键与 `run_config_v3.json` 一致，有单测锁定），
  所以日常不需要挂 `--config`。
- 演练 / 试跑请显式换输出目录：`--output-dir demo_out`（默认输出目录就是最终产出目录）。
- 只想重跑 COT 用 `--config file_processor/run_config_cot_sft.json`（以既有 SFT 问答对为素材）。
- 初版 400 份与 v2 语料已删除（曾是测试用例），`run_config.json` / `run_config_v2.json` /
  `run_config_ocr.json` 随之移除。


---

## 数据处理流程

```
原始语料文件（最终语料 v3: 24 份）
      │
      ▼
┌─────────────────────────── 读取（带磁盘缓存）───────────────────────────┐
│ 文字PDF → pymupdf/pypdf    扫描PDF → 渲染 300DPI → Qwen-VL OCR          │
│ PNG 截图 → Qwen-VL OCR     DOCX → python-docx    HTML → BeautifulSoup  │
└──────────────────────────────────┬────────────────────────────────────┘
                                   ▼
                     清洗（控制字符 / 空白 / 断行 / 乱码行 / 标点）
                                   ▼
                     启发式质量过滤（长度 / 信息密度 / 乱码 / 空表单熔断 / SimHash 去重）
                                   ▼
                     LLM 质量评分（deepseek-flash，并发，单项≥3 且总分≥18：阈值为运营参数，代码默认 20、
                      run_config 实测调至 18，依据见「实测规模」一节）
                                   ▼
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
  预训练（分块 + 元数据）      指令微调（模板 / LLM）        COT（推理链提取）
```

---

## 实测规模（400 文件全量，LLM 质检开启）

### 批次 A：250 文件（文本 PDF / Word / 源码 HTML / 截图无关项 / TXT）

| 指标 | 数值 |
|------|------|
| 清洗+去重 | 250 → 248（2 个真实模板重复件被 SimHash 拦下） |
| 质检通过 | 150/248（60.5%），评分 8 并发约 4 分钟 |
| 质检均分 | 完整性 4.12 / 语言 3.81 / 密度 4.52 / 结构 3.99 / 可读 3.92 |
| Token 估计 | 3,873,089 |

### 批次 B：150 文件（扫描版 PDF ×100 + 截图 HTML ×50，Qwen-VL OCR）

| 指标 | 数值 |
|------|------|
| 读取/OCR | 150/150 成功，0 失败；约 1580 页 PDF + 50 张截图逐页识别 |
| 耗时 | 首段 18.5 分钟完成 73 文件（含暂停恢复），全部读取缓存后补跑仅 3.3 分钟 |
| 质检通过 | 94→98/150（含补跑救回限流件；1 个 PNG 被内容审核误判 HTTP 400，属平台侧行为） |
| Token 估计 | 1,713,131 |

### 合并终态（`--resume` 追加，`verify_datasets.py` 全绿）

| 数据集 | 记录数 | 来源数 | 大小 |
|--------|--------|--------|------|
| pretrain | 7384 | 277 | 10.99 MB |
| instruction | 2455 | 242 | 3.57 MB |
| cot | 175 | 148 | 0.37 MB |

- **input 语义修复后口径**：instruction 73%、cot 90% 的记录 `input` 携带作答所需材料（开卷），
  留空者全部是题面自包含的样本（定义式通识问答 / 因果条件推理）；指令/推理样本量下降是
  主动清除"闭卷背诵原文"式坏样本的结果（原 cot 645 条多为章节编号误判与无材料幻觉引导）；

- 零精确重复 / 零字段缺失 / 零 U+FFFD / 分块索引一致；
- 入选语料语言分布（来源级）：zh 185 / en 39 / mixed 52 / unknown 1；Token 合计 ≈ 3.9M；
- 质检通过率演进：8.1%（抽样自伤误判）→ 61.3%（修复后，阈值 20）→ **69.6%**
  （277/398，阈值调至 18 后重评淘汰集：146 篇救回 29 篇，pretrain 记录 +38%）；
- 阈值 20→18 的依据：离线重放显示通过组总分均值 20.4 贴线，淘汰主体不是垃圾
  而是"良"档；单项 ≥3 底线保留，淘汰者仍以真实缺陷为主（OCR 碎断、空洞模板）；
- 空表单申报书（`form_template_grid`）由启发式层熔断，阈值标定见排错要点；
- 无质检基线（同 250 文件）：pretrain 8468 / instruction 1492 / cot 715，供对照
  （派生两类是旧转换器口径，仅 pretrain 与当前可直接对比）。

产出完整性可用 `python verify_datasets.py` 自检（记录数/metadata 一致性、
精确重复、字段缺失、U+FFFD、分块索引、丢弃原因直方图）。

---

## 输出说明

```
datasets_v3_20260927/          # 实际目录名由 output_dir 决定（本轮 = datasets_v3_20260927）
├── pretrain/
│   ├── data.jsonl
│   └── metadata.json          # 记录数 / 来源数（累计口径）/ 本次新增来源 / 生成时间 / 关键配置
├── instruction/
│   ├── data.jsonl
│   └── metadata.json
├── cot/
│   ├── data.jsonl
│   └── metadata.json
├── quality_report.json        # 各维度均分、通过率、失败原因 TOP10、失败来源清单
├── run_report.json            # 本次运行统计（含脱敏后的配置快照）
└── rejected.jsonl             # 被丢弃文本（source/stage/reason/正文片段）
```

数据格式：

```jsonc
// pretrain
{"text": "分块后的文本…", "meta": {"source": "xxx.pdf", "type": "pdf_download",
  "language": "zh", "chunk_index": 0, "chunk_total": 17,
  "size_bytes": 1496, "token_count_est": 332}}

// instruction (SFT)——开卷：答案来自特定文档的，材料必须放 input
{"instruction": "根据以下关于「数据处理的流程」的材料，总结其主要内容的要点。",
 "input": "数据处理通常包含多个步骤…", "output": "1. …\n2. …", "source": "article.txt"}
// instruction——闭卷合规形态：题面自包含的通识定义样本才允许空 input
{"instruction": "什么是不正当竞争行为？", "input": "",
 "output": "本法所称的不正当竞争行为，是指经营者在生产经营活动中…", "source": "law.txt"}

// cot——步骤/清单类开卷，材料进 input
{"instruction": "根据以下材料，清点其中总共列出了几个条目？请逐条核对。",
 "input": "1. …\n2. …\n3. …", "chain_of_thought": "逐条清点材料：…",
 "output": "材料共列出3个条目。", "source": "doc.pdf"}
// cot——因果/条件推理：已知条件已写进题面，input 留空自洽
{"instruction": "根据已知信息，分析训练数据存在噪声会得出什么结论？请给出推理过程。",
 "input": "", "chain_of_thought": "1. 已知条件…", "output": "模型需要正则化…", "source": "doc.txt"}
```

---

## 质量评分

清洗后的文本逐条交给 `deepseek-flash` 打分（并发执行，`--max-workers` 控制并发数）：

| 维度 | 范围 | 说明 |
|------|------|------|
| 内容完整性 | 1-5 | 有完整开头结尾，未被截断 |
| 语言规范性 | 1-5 | 无乱码、无 OCR 错误、语法通顺 |
| 信息密度 | 1-5 | 有实质内容，非空泛重复 |
| 结构清晰度 | 1-5 | 段落层次分明 |
| 总体可读性 | 1-5 | 语言流畅、逻辑连贯 |

**阈值**：默认单项 ≥ 3 且总分 ≥ 20/25（可用 `--quality-min-per-dim/--quality-min-total` 调整；
两份 `run_config*.json` 实跑采用 **18**——离线重放发现通过组均值 20.4 贴线、18–19 档存在
大量"良"级文档，降低后重评救回 29 篇、pretrain 记录 +38%，单项 ≥3 底线保留）。

**长文档抽样送评**：超过约 3700 字的文本按 开头/中部/结尾 三段各 1200 字送评，段间以
`【……中间内容因长度限制省略……】` 标记连接，提示词明确该标记不视为截断（见排错要点）。

- **API key 来源**：真实环境变量 `DASHSCOPE_API_KEY` > 项目根 `.env`（自动加载，见快速开始）；两者都没有时自动跳过评分（仅走启发式过滤）并打印提示；
- 因 API 错误（key 失效/配额）被淘汰的条数会在报告 `api_errors` 中单独统计，避免误判为数据质量问题。

---

## OCR 方案

扫描版 PDF 与 PNG 截图统一走 **Qwen-VL-Plus**（DashScope，国内直连）：

- 扫描 PDF 渲染为 300 DPI PNG 后逐页识别（`ocr_dpi` 可调），识别结果保留段落与表格结构；
- 8 并发（`max_workers`），每页约 10–20 秒；
- 读取结果进入缓存，`--resume` 重跑不会重复计费；
- 未配置 API key 时，扫描版 PDF 会明确报错并计入失败，而不是静默尝试加载本地模型。

---

## 命令行参考

| 参数 | 说明 |
|------|------|
| `--config PATH` | JSON 配置文件（CLI 显式参数优先于配置） |
| `--input-dir DIR` | 根目录，自动映射 `文本pdf/版式pdf/word/源码html/截图html/txt` 等子目录 |
| `--files a,b,c` | 只处理指定文件 |
| `--output-dir/-o` | 输出目录（默认 `datasets`） |
| `--output-types` | `pretrain,instruction,cot` 的子集 |
| `--chunk-size/--chunk-overlap` | 预训练分块（`0` = 不分块） |
| `--min-text-length` | 最小文本长度（默认 30） |
| `--dedup-threshold` | SimHash 文档判重阈值（默认 0.90，越高越保守） |
| `--max-samples` | 每类数据集样本上限（0 = 不限） |
| `--resume` | 保留已有输出；已产出/确定性淘汰的来源在读取前即跳过，质检淘汰来源自动重试 |
| `--no-cache` / `--clear-cache` | 禁用 / 清空读取缓存 |
| `--no-quality` | 即使有 API key 也跳过 LLM 评分 |
| `--no-rejected` | 不生成 `rejected.jsonl` |
| `--max-workers` | 读取与评分并发数（默认 4） |
| `--dry-run` | 只打印配置与待处理文件数 |
| `--llm --llm-provider dashscope` | 启用 LLM 生成指令/COT |
| `-v/--verbose` | DEBUG 日志 |

---

## 测试

```bash
python -m unittest discover -s tests -v
```

覆盖：配置解析与 key 脱敏、`.env` 加载器（shell 优先/重复键先到先得/缺文件静默/路径指向项目根）、
输出布局与重建/续跑语义、样本级精确去重、空表单熔断（空格/复读格两种触发+名单豁免+短表豁免）、读取缓存失效、
中文标点保护、编号步骤/列表换行保留、质量过滤误判回归、去重指纹稳定性（金样例）与
bigram/长度比标定语义、质检抽样策略（头/中/尾三段、保留真实结尾）与响应解析、
各读取器（文字/扫描 PDF 分流、OCR 临时目录清理、GBK 编码、HTML 类型判定与
行内链接不切碎段落、wiki 皮肤容器剥离）、清洗器（界面残留/页脚/Category 标签清除）、
三种格式转换、pipeline 端到端（含重跑不重复、resume 读前跳过与确定性丢弃集、
metadata 累计来源口径）、**样本字段契约**（开卷样本必带材料、自包含样本留空、
定义清洗门控拦截编号粘连/半截答案、论文章节编号不得伪装成流程步骤）。
测试不需要 API key，也不写工作区之外的文件。

---

## 设计与排错要点

- **中文标点保护**：Unicode 标准化使用 NFC 而非 NFKC（NFKC 会把 `，` 折叠成 `,`），全角→半角只在英文文本上执行。
- **长文档不整篇丢弃**：`max_text_length` 默认 1000000，60KB+ 的论文交由分块处理，不再因长度被淘汰。
- **乱码判定用比例**：不再使用"单字符出现 >100 次即乱码"（中文高频字会被误杀），改用连续重复 + 单字符占比。
- **质检送评文本必须自洽**：旧实现送 `text[:3000]` 而提示词又规定"句中截断→完整性≤2"，
  等于系统性自我否决：实测 250 篇通过率被压到 8.1%（221 篇 low_score，完整性均分 2.2）。
  改为头/中/尾三段抽样 + 省略标记 + 提示词区分"抽样断开"与"真实截断"（结尾段始终取文档
  最后 1200 字，真实残缺仍能识别）。修复后同一语料通过率 60.5%，完整性均分 4.13。
- **HTML 行内链接不得切碎段落**：旧实现把每个文本节点当独立段落，`本文是<a>法律法规</a>。`
  被抽成三行，结构/可读性维度全毁。现按 块级/行内 区分缓冲，CJK 相邻片段自动去连接空格。
  同时剥离 MediaWiki 皮肤容器（`mw-jump-link`/`mw-dismissable-notice`/`printfooter`/`catlinks`/
  `licensetpl`/`mw-list-item`/`vector-sticky-header` 等）并排除注释节点（parser-cache 注释会以
  NavigableString 子类身份混进正文）。清洗器再兜底清除并入段落的皮肤文案与 `Category:` 标签。
- **空表单熔断（零误伤标定）**：空白申报/审批/推荐表格模板（主体是 `姓名 | | 性别 | |`
  空网格或 `身份证号 | 身份证号 | 身份证号` 合并复读）不适合训练且白耗质检 API，
  在启发式过滤层拦截（`form_template_grid`）。规则 = 长度≥1000 且 管道行字符占比≥0.45
  且 叙述文本占比≤0.30 且（空格子≥0.5 或 复读格子≥0.4），在 400 文件真实语料上标定：
  248 个已通过文档**零误伤**、精准命中 4 篇空表单；填了真实数据的名单/统计表（管道占比
  0.9+ 但格子填满、含长实体名）不受影响——单靠管道占比区分不开，空/复读格子才是判据。
- **通过率不是越高越好**：60% 一度看起来偏低，逐特征复盘（断行率/符号垃圾/结尾完整性/
  叙述占比 对照通过组分布）后确认约八成淘汰是真实缺陷（OCR 碎断、空洞模板、半截文档），
  可安全动作只有边界档回收（阈值 20→18）与垃圾前置（表单熔断）——都做了，终值 69.6%。
- **`input` 字段语义（开卷/闭卷契约）**：答案来自某篇具体文档的样本（标题阅读、
  列表/编号条目提取、步骤流程）必须把材料放进 `input`——否则等于要求模型闭卷背诵
  一篇没见过的文档，是在教幻觉；定义式通识问答、"因为…所以"式因果推理等题面
  自包含的样本留空才合规。这类错误 `verify_datasets.py` 查不出来（它校验键存在性，
  格式合法 ≠ 语义正确），由 18 个转换器契约测试锁死。配套清洗守卫：定义式剥离
  编号前缀/术语长度上限/动词误接黑名单/句末标点截断（防"答案吞进第十八条"）；
  编号块按 +1 单调切段取最长、描述中位长度、英文论文节名守卫（防目录碎片伪装成
  "40 步流程"）；所有样本的作答线索截断在材料预算之内（防 output 引用 input 之外的内容）。
  转换器改动后用 `python rebuild_derived.py` 从读取缓存免费重建派生数据集（零 API，
  质检门槛与 pretrain 不动）。
- **去重标定（实测 250 文件）**：
  - 指纹必须用稳定内容哈希（blake2b 64 位）。旧实现用内置 `hash()`，其进程随机盐导致同一语料两次运行分别判重 12 篇/8 篇（交集仅 4 篇），结果完全不可复现。
  - token 用 **bigram**（中文相邻字 + 英文相邻词）。旧实现中文单字/英文单词在同领域语料上误杀严重：白皮书 18/50、arXiv 论文 12/50。
  - 只清"近乎复制"：相似度 ≥ 0.90 且长度比 ≥ 0.6。同模板不同填充的文档（不同高校简介页 0.906/0.891、白皮书不同主题系列 0.78-0.83）全部保留；真正的模板重复件（两份申报书 0.938）被识别。
  - 样本级兜底：`JSONLFormatter` 在写入时跳过与同一数据集已有记录**完全相同**的记录（本次实测拦下 37 条跨来源重复指令样本），`run_report.json` 中计为 `exact_duplicate_samples_skipped`。
- **输出重建与续跑**：每次运行默认清空目标 `data.jsonl`；`--resume` 改为追加，并且**跳过判定前移到读取之前**——跳过集 =（各数据集已产出来源 ∪ 上轮 `rejected.jsonl` 中 clean/filter 阶段的确定性丢弃来源）。被跳过者不再读取/评分/转换；`quality` 阶段淘汰（限流、瞬时 API 错误、内容审核误伤）仍会重试，实测重跑可救回约 4/5。
  注意"确定性丢弃"必须一起跳过：SimHash 判重依赖同轮上下文，近重复文档单独重跑时会因相似对象缺席而放行、污染数据集（该 bug 由 resume 前移测试暴露并修复）。resume 时 `rejected.jsonl` 为**追加**写，保留历史判据。
- **metadata.sources 是累计口径**：resume 补跑只写本次几个来源时，旧实现会把累计 248 覆盖成 4；现在写累计值并另记 `sources_this_run`，`verify_datasets.py` 增加二者一致性校验（历史产出已用一次性脚本修正为 248/218/220）。
- **读取缓存位置**：`.cache/read_cache.jsonl`（键为绝对路径 + mtime + size，文件变化自动失效；只缓存成功结果）。
- **临时目录**：扫描 PDF 渲染页写入 `ocr_temp/`（`ocr_work_dir` 可配），运行后自动清理；**中断运行会残留大量 PNG**，已被 `.gitignore` 忽略，可随时整个删除（OCR 文本结果另存读取缓存 `.cache/read_cache.jsonl`）。
- **日志**：后台运行建议 `cmd /c "python -m file_processor.main ... > ocr_runN.log 2>&1"`（pwsh 的 `>` 会写出 UTF-16 造成乱码）；`check_progress.py` 自动选择最新的 `ocr_run*.log`，也可 `python check_progress.py 指定日志` 覆盖。
- **限流救回**：若因 API 偶发限流出现 `Max retries exceeded`，直接再跑一次 `--resume` 即可（只重试质检淘汰来源，其余全部跳过）。

---

## 数据来源与重建

数据来源、采集口径、校验结果与重建命令见 [`data/README.md`](data/README.md)。
数据构建工具链在 `data/_pipeline/`，重跑数据时按该文档执行即可。

历史遗留输出（旧版追加写导致的重复样本堆积、各轮中断/作废产出、demo 输出、OCR 临时件）已于 2026-09-27 统一清理，约释放 500MB；需要时重跑重建即可（数据本身在 `data/`）。

---

## 本次优化记录

| 问题 | 处理 |
|------|------|
| 明文 API key（`run_quality_score.py`） | 移除，改为环境变量/`--api-key`；该脚本默认不再覆盖原文件（`--in-place` 才覆盖且自动备份） |
| 追加写导致跨次运行重复样本 | `JSONLFormatter` 每次运行重建；新增 `--resume` 续跑语义；旧输出归档到 `_legacy_flat/` |
| README 与实际目录/路径不一致 | 输出改为 `<output_dir>/<type>/data.jsonl`（与文档一致），README 全量重写 |
| 高频汉字被误判乱码 | `_is_garbage` 改为连续重复 + 单字符占比；实测真实白皮书由"丢弃"变为"通过" |
| 60KB+ 论文因长度整篇丢弃 | 默认 `max_text_length` 提升到 1000000，长文交给分块 |
| NFKC 破坏中文标点、列表/步骤被合并 | 改用 NFC；结构化行保留换行（实测 COT 样本 6 → 12） |
| 无缓存、OCR 无法续跑 | 新增 `utils/cache.py` + `--resume`（实测续跑 6/6 命中缓存、0 重复） |
| 质量评分串行、慢 | 改为并发（`max_workers`），新增 `api_errors` 统计与失败来源清单 |
| 被丢弃文本不可追溯 | 新增 `rejected.jsonl` + `run_report.json`（配置自动脱敏） |
| 死代码：本地 OCR 方案、测试图、无关输出 | 归档到 `_archive/` 并附说明；`hf_download_model.py` 按 huggingface_hub ≥1.0 修正 |
| 无依赖清单 / 无 .gitignore / 无测试 | 新增 `requirements.txt`、`.gitignore`、74 个测试；仓库完成首次提交 |
| 脚本输出乱码 | 后台脚本统一 UTF-8 输出 |
| `chunk_overlap > chunk_size` 直接崩溃 | 自动收敛 overlap，`--chunk-size 100` 也能正常跑 |
| SimHash 用 `hash()` 随机盐，判重结果跨进程漂移 | 改 blake2b 稳定 64 位哈希 + 金样例回归测试 |
| 去重误杀：中文单字/英文单词 token 在同领域语料上区分度不足（实测最高 18/50、12/50） | 改中英 bigram + 阈值 0.90 + 长度比≥0.6 约束，实测 250 文件仅 2 个真实模板重复被清 |
| 同一数据集内出现完全相同的样本记录（实测 37 条） | `JSONLFormatter` 写入时按规范化指纹去重（含 resume 历史指纹），`run_report.json` 统计跳过条数 |
| 质检只送前 3000 字且提示词惩罚"截断"，长文被系统性误杀（通过率 8.1%） | 头/中/尾三段抽样 + 省略标记 + 规则区分"抽样断开/真实截断"，通过率恢复至 60.5%，完整性均分 2.2→4.13 |
| wiki 下载件带导航横幅、页脚模板、`Category:` 标签；注释文本混入正文；行内链接把句子切成三段 | HTML reader 剥离皮肤容器 + 排除注释节点 + 块级缓冲重组段落；TextCleaner 增加 UI 行/短语级清除（真实文件实测零残留） |
| 质检评分限流时只留 "Max retries exceeded" 无法诊断 | 保留末次真实错误信息、重试 3→4 次；250 批 api_errors 6→1 |
| key 依赖手动 export / setx（明文进注册表） | 新增零依赖 `.env` 自动加载（shell 优先、gitignore 覆盖、值不落日志）+ `.env.example` 模板 + 6 个单测 |
| `--resume` 跳过判定在输出阶段：已通过来源被重复评分（烧 API） | 判定前移到读取之前：读/评/写全免；质检淘汰来源仍重试（限流可救回） |
| resume 单独重跑近重复文档会因判重上下文丢失而放行（新跳过逻辑暴露的隐患） | 跳过集并入上轮 clean/filter 确定性丢弃来源；rejected.jsonl 改追加写保留历史判据 |
| resume 补跑把 `metadata.sources` 从 248 覆盖成本次的 4 | 写累计口径并新增 `sources_this_run`；`verify_datasets.py` 增加一致性校验；历史值已修复 |
| `check_progress.py` 写死 `ocr_run.log`（实际日志名带批次后缀） | 自动选择最新 `ocr_run*.log`，支持位置参数显式指定 |
| 空白表单/网格模板混进质检候选（白付费 + 拉低平均） | `QualityFilter._is_form_template` 启发式熔断，400 文件标定零误伤；4 个单测 |
| 质检阈值 20 过严：通过组总分均值 20.4 贴线，18–19 档"良"级文档整批被拒 | run_config 阈值调 18（单项≥3 底线保留），`--resume` 自动只重评淘汰集：救回 29/146，pretrain 5345→7384 条（+38%），通过率 61.3%→69.6%，成本 <1 元 |
| sft/cot 的 `input` 恒为空：所有模板硬编码 `""` 从未接线；标题模板 output 恒 None 是死代码；列举/步骤样本让模型闭卷背诵原文（教幻觉） | 按开卷/闭卷契约重建两个转换器（阅读/列举/步骤材料进 input，定义/因果自包含留空），删死代码与"解释「条目」的含义"同义反复模板；定义清洗门控+编号块守卫+材料预算一致性；`rebuild_derived.py` 本地免费重建：instruction 2329→2455（73% 带材料）、cot 645→91（清掉的全是章节误判/语义错配噪声），新增 11 个契约测试锁死字段语义 |
| cot 召回缺口（91 条定界后复查）：`一、/（一）` 中文编号清单不被 COT 清点通道识别（instruction 列举通道已验证的守卫未回填）；因果关联词停留在最初 3 组，"由于…因此/只有…才"等真实推理整批丢弃 | 新增中文编号清点通道（+1 中文序数链、2~12 条、中位长≥12 严一档防"一、短标题"目录；只产清点不产流程）；因果扩到 8 组关联词并配自包含守卫（同句硬约束、拒句读/分号/引号残迹、拒悬空指代与叙事式前提）。`rebuild_derived.py` 免费重建：cot 91→175（来源 86→148，因果 8→16），计数-材料一致性/题面自包含/卫生三审计 0 误报，+7 契约测试（109→116） |