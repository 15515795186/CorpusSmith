# utils/ — 跨层工具

四个零业务依赖的小模块，被 cleaners / converters / pipeline 复用。

## lang_detect.py — 语言检测与 token 估算

- `detect_language(text)`：取前 2000 字，统计 中文字符/英文字母/数字 占"有效字符"比例：
  zh>0.6→`zh`，en>0.6→`en`，双>0.2→`mixed`，否则 `unknown`。
  结果驱动三处行为：清洗的标点策略、转换器模板语言（`contains_english(标题)`）、
  pretrain meta 与统计报告的语言分布。
- `estimate_token_count`：中文按 **1.5 token/字**、其余字符 0.4/字；英文 0.35/字符。
  只用于 `token_count_est` 估算（预算参考，非计费口径）。

## dedup.py — SimHash 近重复去重（文档级）

- `SimHash(64 位)`：token 一律 **bigram**——英文相邻词对（`[a-zA-Z]{2,}` 小写）
  + 中文相邻字对。
  标定教训：英文 unigram 把同领域不同论文判重（实测 12/50 误杀），中文单字同理（18/50）。
- `_hash`：**blake2b(digest_size=8) 稳定 64 位**。内置 `hash()` 有进程随机盐，
  旧实现同一语料两次运行判重结果完全不同（12 篇/8 篇，交集 4），不可复现。
- `Deduplicator.is_duplicate(text)`：
  - <50 字不参与（指纹无意义）；只取前 5000 字算指纹；token<20 不判重；
  - 与历史每个 (指纹,长度) 比较：**相似度 ≥0.90 且 长度比 ≥0.6** 才判重——
    只清"近乎复制"，同模板不同填充（不同高校简介页 0.906/0.891）全部保留；
  - 判重依赖同轮上下文：这就是 resume 跳过集必须并入"确定性丢弃来源"的原因
    （近重复文档单独重跑会因参照物缺席而放行）。
- `reset()`：pipeline 每轮开始时清空历史集合。

## cache.py — ReadCache（读取结果磁盘缓存）

- 键 = `绝对路径|st_mtime_ns|st_size`：**源文件一改自动失效**，无需手动清理。
- 存储 = JSONL 逐条追加（`.cache/read_cache.jsonl`），每条立即落盘——
  OCR 跑一半崩溃不丢已完成页面；加载时忽略损坏行。
- **只缓存成功结果**：API 抖动失败不固化，下次运行自然重试。
- 线程安全（`threading.Lock`），`--clear-cache` 调 `clear()` 删文件。
- 存的是**清洗前**的原始读取文本：清洗器/转换器改规则后 `--resume` 重放即可生效，
  不必清缓存（`rebuild_derived.py` 正是建立在这一性质上）。

## stats.py — StatsCollector

一个计数字段集合（文件数/类型分布/成败/缓存命中/过滤数/样本数/语言分布/token 合计/
错误尾巴），`print_summary()` 输出控制台报告；机器可读版本由
`pipeline._write_run_report()` 写 `<output_dir>/run_report.json`。

## 依赖

- 内部：`stats → lang_detect`；`dedup`、`cache`、`lang_detect` 仅用标准库
- 被引用：`quality_filter.py`（lang_detect+dedup）、三个转换器（lang_detect）、
  `pretrain_converter`（lang_detect）、`pipeline.py`（cache+stats+lang_detect）
