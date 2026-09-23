# file_processor/ — 核心包

3504 行 / 32 个 py。六层流水线 + 根目录五个编排文件。
各子目录另有 README（readers / cleaners / converters / formatters / llm / utils），
本文件讲**包根文件**与**整条数据流**。

## 目录结构与依赖总览

```
file_processor/
├── main.py            CLI 入口：argparse → build_config → Pipeline.run()
├── config.py          ProcessorConfig（dataclass）：唯一配置真源
├── dotenv.py          零依赖 .env 加载器（main 在读环境变量之前调用）
├── pipeline.py        编排器 FileProcessorPipeline（本包的"主函数"）
├── quality_scorer.py  LLM 文档质检（qwen-plus，直连 dashscope SDK）
├── run_config.json    批次 A 运行配置（250 个可提取文本的文件）
├── run_config_ocr.json批次 B 运行配置（150 个需 OCR 的文件）
├── readers/    ──┐
├── cleaners/     ├─ 被 pipeline 组合（依赖方向单向：pipeline → 各层，层间不互调）
├── converters/   │
├── formatters/ ──┘
├── llm/          仅 converters 可选注入（默认关闭）
└── utils/        被 cleaners/converters/pipeline 复用
```

依赖速查（谁 import 谁）：

| 使用者 | 依赖 |
|---|---|
| `pipeline.py` | config, readers(全部5种+base), cleaners(3), converters(3), formatters, utils.cache/stats/lang_detect, quality_scorer, llm(按 provider) |
| `main.py` | config, pipeline, formatters（dry-run 计数）, utils.cache（--clear-cache）, dotenv |
| `quality_scorer.py` | 仅标准库 + `dashscope`（不依赖包内其他模块，可单独测试） |
| `cleaners/quality_filter.py` | utils.lang_detect, utils.dedup |
| `converters/*` | readers.base.ReaderResult, utils.lang_detect, （可选）llm 实例由外部注入 |
| `formatters/jsonl_formatter.py` | 无包内依赖 |
| `readers/pdf_reader.py` | readers.ocr_reader（组合，非继承） |

## 一次运行经过哪些代码（run() 顺序）

```
main.build_config()
  dotenv.load_dotenv()                    # .env → os.environ（已有真实环境变量优先）
  ProcessorConfig.from_json(run_config)   # JSON 覆盖默认值
  CLI 参数只覆盖"显式传入"的项             # 未传的保持 JSON/默认值
FileProcessorPipeline.run()
  1  _scan_files()          glob 各 input_dirs，按路径排序（可复现性前提）
  1b Resume 前置跳过         跳过集 = formatters.read_sources(∪各数据集)
                            ∪ rejected.jsonl 中 clean/filter 阶段确定性丢弃来源
  2  _read_all()            并发读：命中 ReadCache 直接返回（OCR 不重复付费）
                            失败 → 只进 stats（files_failed/errors），不进 rejected、不占后续成本
  3  _clean_all()           html 来源先 HTMLCleaner 再 TextCleaner；其余仅 TextCleaner
                            空文本 → rejected(stage=clean, reason=empty_after_clean)
  4  QualityFilter.check()  启发式（长度/密度/乱码/语言/空表单/SimHash 去重）
                            不过 → rejected(stage=filter)
  5  _quality_check()       QualityScorer.score_batch() 并发 LLM 评分
                            单项≥quality_min_per_dimension 且 总分≥quality_min_total
                            不过 → rejected(stage=quality)（API 错误也记此阶段，resume 会重试）
  6  _convert_and_output()  三种转换器逐文档产出 → JSONLFormatter 指纹去重写入
                            写 metadata.json（sources 累计 + sources_this_run）
  7  _write_run_report()    datasets/run_report.json（to_dict(redact=True) 脱敏配置）
```

## 根目录文件说明

### main.py（CLI）
- `map_input_dirs()`：`--input-dir 根目录` 自动映射已知中文子目录
  （文本pdf/版式pdf/word/源码html/截图html/txt…只收真实存在的）。
- 参数分层：`--config` JSON 打底 → CLI 显式项覆盖 → 环境变量兜底
  （`DASHSCOPE_API_KEY`/`OPENAI_API_KEY` 等，`.env` 已由 dotenv 注入环境）。
- `--dry-run` 只打印配置摘要与文件计数，零写入。

### config.py（ProcessorConfig）
- dataclass 单一真源；`from_json()` 只接收已定义字段（拼错键直接暴露）。
- 关键默认值都有注释解释历史教训（如 `max_text_length=1_000_000`：
  旧默认 32768 曾把 60KB+ 论文整篇误杀，长文应交给分块）。
- `to_dict(redact=True)`：`run_report.json`/日志只出现 `***redacted***`。
- `api_key()`：配置优先，其次 `DASHSCOPE_API_KEY` 环境变量。

### dotenv.py
- 固定读**项目根 `.env`**（包目录上一级，`utf-8-sig` 兼容 BOM）；手工解析
  `KEY=VALUE`，支持 `#` 注释、`export ` 前缀、单/双引号值；
- **只填充"未设置或为空"的环境变量**——shell 里临时 export 的值优先，不被静默覆盖；
- 返回值/日志只出现键名，值任何情况下不落日志；文件不存在静默跳过（.env 是可选项）。

### quality_scorer.py（付费 API 的第二处，也是决策最重的一处）
- 提示词定义 5 维各 1-5 分（完整性/语言/信息密度/结构/可读性），
  要求模型只输出 JSON；`_parse_response()` 容忍围栏与前后杂文。
- **三段抽样 `_build_sample()`**：>3700 字的文档送评"头 1200 +【中间省略标记】+
  中 1200 +【省略】+ 尾 1200"。旧实现送 `text[:3000]` 而提示词惩罚"句中截断"，
  等于系统性自我否决（实测通过率被压到 8.1%）；现在省略标记是明示的抽样行为，
  只有结尾段真实停在句中才算截断（尾部恒取文档最后 1200 字，真实残缺仍能识别）。
- 失败处理：HTTP 非 200/解析失败 重试 `max_retries=4` 次（指数退避），
  `QualityScore.error` 保留**末次真实错误信息**（旧版只剩 "Max retries exceeded"
  无法诊断）；API 失败记 passed=False 但 stage=quality → resume 会重试（限流可救回）。
- `score_batch()`：ThreadPoolExecutor 并发、结果按输入顺序回填；
  `generate_report()` 写 quality_report.json（各维均分、通过率、原因 TOP、失败清单）。

### pipeline.py 的三个关键设计（细节看代码注释）
1. **确定性**：排序扫描 + 稳定哈希 + 模板转换 ⇒ 同输入重跑逐字节一致；
2. **成本闸门顺序**：免费步骤（读缓存命中/启发式过滤）永远排在付费步骤
   （OCR、LLM 评分）之前，被启发式拦下的文档不产生任何 API 费用；
3. **可追溯**：每个丢弃决定写 `rejected.jsonl`（source/stage/reason/正文前 5000 字），
   resume 追加写保留历史判据——跳过集正是靠它区分"确定性丢弃（不再重试）"
   与"质检失败（值得重试）"。

## 测试入口

各层测试在 `../tests/`（test_readers/test_cleaners/test_converters/test_pipeline/…），
`python -m unittest discover -s tests` 全离线、不需要 API key。
