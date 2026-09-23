# formatters/ — 输出落盘层

`jsonl_formatter.py`：数据集唯一的写入口。所有"重建/续跑/去重/元数据"的
语义集中在这一层，转换器不直接碰文件。

## 输出布局

```
<output_dir>/
├── pretrain/data.jsonl + metadata.json
├── instruction/data.jsonl + metadata.json
├── cot/data.jsonl + metadata.json
└── quality_report.json          # get_report_path()，由 pipeline 写入
```

## 核心行为

### 生命周期：prepare → write_batch → write_metadata
- `prepare(name)`：建目录；`overwrite=True`（默认）时**删除旧 data.jsonl**——
  历史教训：追加写导致多次运行的样本跨批次堆叠出大量重复。
- `overwrite=False`（即 `--resume`）：保留旧文件，并把**历史记录的指纹全部预载**
  进 `_seen[name]`，保证新记录不与历史样本重复。

### 样本级精确去重
- `_record_key()`：整条记录 `sort_keys=True` 序列化后的 **blake2b-16 hexdigest**。
  必须稳定哈希（内置 `hash()` 有进程随机盐，跨运行不幂等）。
- `write_batch()` 跳过已见指纹并累加 `duplicates_skipped`
  → `run_report.json` 的 `exact_duplicate_samples_skipped`（实测拦下 37 条跨来源重复指令样本）。

### 查询接口（供 --resume 与报告使用）
- `count_lines(name)`：现读文件数行数（metadata 的记录数来源，不依赖内存计数）。
- `read_sources(name)`：扫已有记录收集 `meta.source`（instruction/cot 记顶层
  `source`）的 **basename 集合**——pipeline 用它构建"已产出来源"跳过集。
- `exists(name)`。

### write_metadata(name, **extra)
固定写 `dataset_name/status/records/data_file`，pipeline 追加
`sources`（累计口径，勿覆盖成本次数）/`sources_this_run`/`generated_at`/
`elapsed_seconds`/`config`（脱敏后）。历史 bug：resume 补跑曾把累计 248 覆盖成 4，
`verify_datasets.py` 现在交叉校验 meta.sources 与实际扫描值。

## 依赖

- 内部：无（只依赖标准库）
- 被 `pipeline.py`、`rebuild_derived.py` 使用
