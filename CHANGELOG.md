# 更新日志

本文件记录本项目的显著变更。
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## Unreleased

### 新增
- `pyproject.toml`：包元数据、依赖与可选依赖组（`pipeline` / `dev`）、ruff 与 mypy 配置，支持 `pip install -e .`
- `constraints.txt`：直接依赖版本锁定
- CI 工作流（`.github/workflows/ci.yml`）：Python 3.10 / 3.12 × Ubuntu / Windows 运行 ruff 与 unittest
- `LICENSE`（MIT）；README 新增「快速开始」章节；`data/README.md` 纳入版本库

### 变更
- **可移植性**：移除机器相关绝对路径——`config.py` 的 `ppl_cache_dir` 默认改为空、
  `scan_ppl_distribution.py` 改用 `PPL_CACHE_DIR` 环境变量、
  `build_independent_chinese_dataset.py` 的浏览器路径改为环境变量 / PATH 探测
- README 的规模与目录结构数字按实际修正，并去除易漂移的硬编码数量
- `.gitignore` 改为白名单策略并补全（`.env.*` 变体、系统文件、工具缓存、`build`/`dist`、`.workbuddy/`）

### 修复
- 删除 `pipeline.py::_write_converted` 中位于 `return` 之后的不可达死代码（残留块引用 10 处未定义变量）
- 清理 14 处未使用导入、2 处死赋值、2 处空 f-string

### 移除
- `_archive/`（已废弃的本地 OCR 方案、测试图与无关输出）

## 0.1.0 - 2026-09-22

### 新增
- 首个可用版本：多格式文档（PDF / Word / HTML / TXT / PNG）读取与 Qwen-VL OCR
- 清洗与去重（SimHash 指纹）、PPL 困惑度预筛闸、LLM 多维度质量评分
- 输出三类数据集：`pretrain`（分块）、`instruction`（SFT）、`cot`（Chain-of-Thought）
- 读取结果落盘缓存与 `--resume` 断点续跑；`run_report.json` / `rejected.jsonl` 可诊断产出
- 零依赖 `.env` 加载器（真实环境变量优先，凭据不入库）
- 标准库 unittest 测试套件（无需 API key）
