# LLM 数据集处理工作流

批量读取 400 份多格式文档（PDF / Word / HTML / TXT / PNG），经 **解析 / OCR → 清洗去重 → LLM 质量评分**，
转换为 **3 种大模型训练数据集**：预训练（pretrain）、指令微调（SFT）、Chain-of-Thought（COT）。

---

## 核心特性

| 特性 | 说明 |
|------|------|
| 增量可复用 | 读取结果（含 OCR）落盘缓存，重跑不再重复解析、不重复调用付费 API |
| 断点续跑 | `--resume` 保留已有输出并跳过已处理来源，OCR 中断后可直接接着跑 |
| 结果可复现 | 文件按路径排序处理；每次运行重建输出，不再出现跨次运行记录堆叠 |
| 可诊断 | 被丢弃文本写入 `rejected.jsonl` 并统计原因，便于定位阈值/OCR 质量问题 |
| 确定性产出 | 每次运行生成 `run_report.json`（文件数、缓存命中、丢弃原因、数据集规模、脱敏配置） |
| 无外部依赖测试 | 53 个单元 + 端到端测试（标准库 unittest），不需要 API key |

---

## 目录结构

```
LLM_Data_Test/
├── file_processor/                  # 核心处理包（32 个 py，约 3.2k 行）
│   ├── main.py                      # CLI 入口
│   ├── pipeline.py                  # 管道编排：扫描→读取→清洗→质检→转换→输出
│   ├── config.py                    # 全局配置（dataclass + JSON 覆盖）
│   ├── quality_scorer.py            # qwen-plus 多维度质量评分（并发）
│   ├── run_config.json              # 运行配置：可提取文本的 250 个文件
│   ├── run_config_ocr.json          # 运行配置：需 OCR 的 150 个文件
│   ├── readers/                     # pdf / word / html / txt / qwen-vl ocr
│   ├── cleaners/                    # text_cleaner / html_cleaner / quality_filter
│   ├── converters/                  # pretrain / instruction / cot
│   ├── formatters/jsonl_formatter.py# 输出布局 + 重建/续跑/元数据
│   ├── llm/                         # 可选：openai / dashscope 生成器
│   └── utils/                       # lang_detect / dedup / stats / cache
├── data/                            # 原始数据 + 数据构建工具链（详见 data/README.md）
│   ├── 文本pdf/                     50 篇英文论文 PDF（文字层）
│   ├── 文本pdf中文版/                50 篇中文论文 PDF（文字层）
│   ├── 版式pdf/                     50 篇英文扫描版 PDF（需 OCR）
│   ├── 版式pdf中文版/                50 篇中文扫描版 PDF（需 OCR）
│   ├── word/                       50 份政务/高校 DOCX
│   ├── 源码html/                    50 个维基法律页面 HTML
│   ├── 截图html/                    50 张页面截图 PNG（需 OCR）
│   ├── txt/                        50 篇政府白皮书 TXT
│   ├── tech_output/                数据构建暂存区
│   └── _pipeline/                  采集/生成/校验脚本（重建数据时使用）
├── datasets/                        # 产出目录（每次运行重建）
├── tests/                           # 单元与端到端测试
├── _archive/                        # 已废弃的本地 OCR 方案（见其 README）
├── check_progress.py                # 进度查看：输入量 / 产出 / 缓存 / 报告 / 日志
├── run_quality_score.py             # 对已有数据集重新评分（不破坏原文件）
├── requirements.txt
└── README.md
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API key（不要写进代码）
export DASHSCOPE_API_KEY="sk-xxx"          # Linux/macOS
setx DASHSCOPE_API_KEY "sk-xxx"            # Windows（新开终端生效）

# 3. 先看配置与待处理文件（不写任何文件）
python -m file_processor.main --dry-run

# 4. 处理 250 个可提取文本的文件
python -m file_processor.main --config file_processor/run_config.json

# 5. 处理 150 个需要 OCR 的文件（耗时较长，建议后台运行）
python -m file_processor.main --config file_processor/run_config_ocr.json

# 6. 中断后接着跑（已处理的文件不会重复调用 OCR API）
python -m file_processor.main --config file_processor/run_config_ocr.json --resume

# 7. 查看进度
python check_progress.py
```

不指定 `--config` 时使用 `config.py` 中的默认路径，即 `data/` 下的 8 个目录（共 400 个文件）。

---

## 数据处理流程

```
400 个原始文件
      │
      ▼
┌─────────────────────────── 读取（带磁盘缓存）───────────────────────────┐
│ 文字PDF → pymupdf/pypdf    扫描PDF → 渲染 300DPI → Qwen-VL OCR          │
│ PNG 截图 → Qwen-VL OCR     DOCX → python-docx    HTML → BeautifulSoup  │
└──────────────────────────────────┬────────────────────────────────────┘
                                   ▼
                     清洗（控制字符 / 空白 / 断行 / 乱码行 / 标点）
                                   ▼
                     启发式质量过滤（长度 / 信息密度 / 乱码 / SimHash 去重）
                                   ▼
                     LLM 质量评分（qwen-plus，并发，单项≥3 且总分≥20/25）
                                   ▼
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
  预训练（分块 + 元数据）      指令微调（模板 / LLM）        COT（推理链提取）
```

---

## 输出说明

```
datasets/
├── pretrain/
│   ├── data.jsonl
│   └── metadata.json          # 记录数 / 来源数 / 生成时间 / 关键配置
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

// instruction (SFT)
{"instruction": "什么是机器学习？", "input": "", "output": "…", "source": "article.txt"}

// cot
{"instruction": "请逐步说明实现过程。", "input": "",
 "chain_of_thought": "1. …\n2. …\n3. …", "output": "结论", "source": "doc.pdf"}
```

---

## 质量评分

清洗后的文本逐条交给 `qwen-plus` 打分（并发执行，`--max-workers` 控制并发数）：

| 维度 | 范围 | 说明 |
|------|------|------|
| 内容完整性 | 1-5 | 有完整开头结尾，未被截断 |
| 语言规范性 | 1-5 | 无乱码、无 OCR 错误、语法通顺 |
| 信息密度 | 1-5 | 有实质内容，非空泛重复 |
| 结构清晰度 | 1-5 | 段落层次分明 |
| 总体可读性 | 1-5 | 语言流畅、逻辑连贯 |

**阈值**：单项 ≥ 3 且总分 ≥ 20/25，否则淘汰（可用 `--quality-min-per-dim/--quality-min-total` 调整）。

- 未配置 API key 时自动跳过评分（仅走启发式过滤），并打印提示；
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
| `--max-samples` | 每类数据集样本上限（0 = 不限） |
| `--resume` | 保留已有输出，跳过已处理来源（配合读取缓存实现真正的续跑） |
| `--no-cache` / `--clear-cache` | 禁用 / 清空读取缓存 |
| `--no-quality` | 即使有 API key 也跳过 LLM 评分 |
| `--no-rejected` | 不生成 `rejected.jsonl` |
| `--max-workers` | 读取与评分并发数（默认 4） |
| `--dry-run` | 只打印配置与待处理文件数 |
| `--llm --llm-provider openai\|dashscope` | 启用 LLM 生成指令/COT |
| `-v/--verbose` | DEBUG 日志 |

---

## 测试

```bash
python -m unittest discover -s tests -v
```

覆盖：配置解析与 key 脱敏、输出布局与重建/续跑语义、读取缓存失效、中文标点保护、
编号步骤/列表换行保留、质量过滤误判回归、三种格式转换、pipeline 端到端（含重跑不重复、resume 命中缓存）。
测试不需要 API key，也不写工作区之外的文件。

---

## 设计与排错要点

- **中文标点保护**：Unicode 标准化使用 NFC 而非 NFKC（NFKC 会把 `，` 折叠成 `,`），全角→半角只在英文文本上执行。
- **长文档不整篇丢弃**：`max_text_length` 默认 1000000，60KB+ 的论文交由分块处理，不再因长度被淘汰。
- **乱码判定用比例**：不再使用"单字符出现 >100 次即乱码"（中文高频字会被误杀），改用连续重复 + 单字符占比。
- **输出重建**：每次运行默认清空目标 `data.jsonl`；需要追加/续跑时使用 `--resume`。
- **读取缓存位置**：`.cache/read_cache.jsonl`（键为绝对路径 + mtime + size，文件变化自动失效；只缓存成功结果）。
- **临时目录**：扫描 PDF 渲染页写入 `ocr_temp/`（`ocr_work_dir` 可配），运行后自动清理。
- **日志**：使用 `ocr_run.log` 记录后台运行日志；`check_progress.py` 会解析其中的 OCR 页面进度。

---

## 数据来源与重建

数据来源、采集口径、校验结果与重建命令见 [`data/README.md`](data/README.md)。
数据构建工具链在 `data/_pipeline/`，重跑数据时按该文档执行即可。

历史遗留输出（旧版追加写导致重复样本堆积）已移至 `datasets/_legacy_flat/`，仅作参考。

---

## 本次优化记录

| 问题 | 处理 |
|------|------|
| 明文 API key（`run_quality_score.py`） | 移除，改为环境变量/`--api-key`；该脚本默认不再覆盖原文件（`--in-place` 才覆盖且自动备份） |
| 追加写导致跨次运行重复样本 | `JSONLFormatter` 每次运行重建；新增 `--resume` 续跑语义；旧输出归档到 `_legacy_flat/` |
| README 与实际目录/路径不一致 | 输出改为 `datasets/<type>/data.jsonl`（与文档一致），README 全量重写 |
| 高频汉字被误判乱码 | `_is_garbage` 改为连续重复 + 单字符占比；实测真实白皮书由"丢弃"变为"通过" |
| 60KB+ 论文因长度整篇丢弃 | 默认 `max_text_length` 提升到 1000000，长文交给分块 |
| NFKC 破坏中文标点、列表/步骤被合并 | 改用 NFC；结构化行保留换行（实测 COT 样本 6 → 12） |
| 无缓存、OCR 无法续跑 | 新增 `utils/cache.py` + `--resume`（实测续跑 6/6 命中缓存、0 重复） |
| 质量评分串行、慢 | 改为并发（`max_workers`），新增 `api_errors` 统计与失败来源清单 |
| 被丢弃文本不可追溯 | 新增 `rejected.jsonl` + `run_report.json`（配置自动脱敏） |
| 死代码：本地 OCR 方案、测试图、无关输出 | 归档到 `_archive/` 并附说明；`hf_download_model.py` 按 huggingface_hub ≥1.0 修正 |
| 无依赖清单 / 无 .gitignore / 无测试 | 新增 `requirements.txt`、`.gitignore`、53 个测试；仓库完成首次提交 |
| 脚本输出乱码 | 后台脚本统一 UTF-8 输出 |
| `chunk_overlap > chunk_size` 直接崩溃 | 自动收敛 overlap，`--chunk-size 100` 也能正常跑 |