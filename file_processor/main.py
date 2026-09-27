#!/usr/bin/env python
"""
LLM 数据集处理工作流入口
批量读取多格式文件 → 清洗 → 过滤(启发式 + PPL 异常闸) → 质量评分 → 转换为 3 种 LLM 训练数据集

前置: 项目根 `.env` 填模型凭据(已被 gitignore)
  DEEPSEEK_API_KEY   文本 LLM —— SFT/COT 生成、质量评分、语义抽检(默认 provider)
  DASHSCOPE_API_KEY  仅 OCR 视觉链(qwen-vl-plus); 文本生成已不依赖它

用法:
  # 主流程: 第三版爬取语料(8 类 × 3 = 24 份) -> pretrain / instruction / cot
  python -m file_processor.main --config file_processor/run_config_v3.json

  # 只从既有 SFT 问答对生成 COT(判类 → 回原文搜据 → 写规范字段)
  python -m file_processor.main --config file_processor/run_config_cot_sft.json

  # 只处理需要 OCR 的文件(截图 / 扫描 PDF)
  python -m file_processor.main --config file_processor/run_config_ocr.json

  # 断点续跑：保留已有输出，跳过已处理来源(中断后接着跑, 不重复调 API)
  python -m file_processor.main --config file_processor/run_config_ocr.json --resume

  # 内置默认路径(data/ 下 8 个目录, 共 400 份初版语料)
  python -m file_processor.main

  # 指定具体文件 / 只输出部分数据集 / 指定输出目录与并发
  python -m file_processor.main --files doc1.pdf,doc2.docx --output-types pretrain
  python -m file_processor.main --files doc1.pdf,doc2.docx --output-dir demo_out --max-workers 2

  # 换模型: provider 默认 deepseek(deepseek-flash); 需要更强推理可换 deepseek-v4-pro;
  #         百炼欠费后文本链已切 DeepSeek, 传 dashscope 仅作回退
  python -m file_processor.main --llm --llm-provider deepseek --llm-model deepseek-v4-pro

  # 无凭据时跑结构链路(跳过质量评分): 加 --no-quality
  python -m file_processor.main --config file_processor/run_config_v3.json --no-quality

产出: <output_dir>/{pretrain,instruction,cot}/data.jsonl + metadata.json,
      以及 run_report.json / rejected.jsonl / review_output*.jsonl(争议样本待人工审核清单)
"""

import argparse
import logging
import os
import sys

# Windows console UTF-8 support
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from file_processor.config import ProcessorConfig
from file_processor.pipeline import FileProcessorPipeline
from file_processor.formatters.jsonl_formatter import JSONLFormatter
from file_processor.utils.cache import ReadCache

# --input-dir 时的子目录自动映射（同时支持中文/英文目录名）
SUBDIR_CANDIDATES = {
    "pdf_download": ("文本pdf", "pdf_download"),
    "pdf_download_cn": ("文本pdf中文版", "pdf_download_cn"),
    "pdf_photo": ("版式pdf", "pdf_photo"),
    "pdf_photo_cn": ("版式pdf中文版", "pdf_photo_cn"),
    "word": ("word",),
    "html_source": ("源码html", "html_source"),
    "screenshot": ("截图html", "html_screenshot", "screenshot"),
    "txt": ("txt",),
}
VALID_OUTPUT_TYPES = ("pretrain", "instruction", "cot")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量文件 → LLM数据集处理工作流",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m file_processor.main
  python -m file_processor.main --input-dir data
  python -m file_processor.main --config file_processor/run_config_ocr.json --resume
  python -m file_processor.main --files a.pdf,b.docx,c.html
  python -m file_processor.main --output-types pretrain,instruction --max-workers 8
        """,
    )

    # 输入（默认 None = 不覆盖配置）
    parser.add_argument("--input-dir", "-i", default=None,
                        help="根输入目录（自动映射已知子目录）")
    parser.add_argument("--pdf-dir", default=None, help="PDF文件目录")
    parser.add_argument("--word-dir", default=None, help="Word文件目录")
    parser.add_argument("--html-dir", default=None, help="HTML文件目录")
    parser.add_argument("--txt-dir", default=None, help="TXT文件目录")
    parser.add_argument("--files", "-f", default=None, help="指定具体文件（逗号分隔）")

    # 输出
    parser.add_argument("--output-dir", "-o", default=None,
                        help="输出目录 (default: datasets)")
    parser.add_argument("--output-prefix", default=None, help="输出文件前缀")
    parser.add_argument("--output-types", default=None,
                        help="输出数据集类型，逗号分隔 (default: pretrain,instruction,cot)")

    # 清洗控制
    parser.add_argument("--min-text-length", type=int, default=None,
                        help="最小文本长度 (default: 30)")
    parser.add_argument("--max-text-length", type=int, default=None,
                        help="最大文本长度，超长整篇丢弃 (default: 1000000)")
    parser.add_argument("--dedup-threshold", type=float, default=None,
                        help="SimHash 文档判重相似度阈值 (default: 0.90)")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="预训练分块大小 (default: 1024)，0=不分块")
    parser.add_argument("--chunk-overlap", type=int, default=None,
                        help="预训练分块重叠 (default: 128)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="每种数据集最大样本数 (0=不限制)")

    # LLM 生成
    parser.add_argument("--llm", action="store_true", help="启用LLM生成指令/COT")
    parser.add_argument("--llm-provider", default=None, choices=["dashscope", "deepseek"],
                        help="LLM供应商 (default: deepseek)")
    parser.add_argument("--llm-model", default=None, help="LLM模型名 (覆盖默认)")

    # 质量评分
    parser.add_argument("--no-quality", action="store_true",
                        help="即使提供了 API key 也跳过 LLM 质量评分")
    parser.add_argument("--quality-min-per-dim", type=int, default=None,
                        help="质量评分单项最低分 (default: 3)")
    parser.add_argument("--quality-min-total", type=int, default=None,
                        help="质量评分总分最低分 (default: 20)")

    # 运行控制
    parser.add_argument("--config", default=None, help="JSON配置文件路径")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑：保留已有输出，跳过已处理来源")
    parser.add_argument("--no-cache", action="store_true", help="禁用读取结果缓存")
    parser.add_argument("--clear-cache", action="store_true", help="清空读取缓存后退出")
    parser.add_argument("--no-rejected", action="store_true",
                        help="不保存 rejected.jsonl（被丢弃文本）")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="并发工作数 (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印配置与待处理文件数，不实际运行")

    # 其他
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")

    return parser.parse_args()


def map_input_dirs(base: str) -> dict:
    """把根目录映射到已知子目录（只保留真实存在的）"""
    mapping = {}
    for key, candidates in SUBDIR_CANDIDATES.items():
        for name in candidates:
            path = os.path.join(base, name)
            if os.path.isdir(path):
                mapping[key] = path
                break
    return mapping


def build_config(args) -> ProcessorConfig:
    """由 JSON 配置 + CLI 参数构建最终配置（CLI 只覆盖显式传入的项）"""
    config = ProcessorConfig.from_json(args.config) if args.config else ProcessorConfig()

    # ---- 输入 ----
    if args.input_dir:
        mapped = map_input_dirs(args.input_dir)
        if not mapped:
            raise SystemExit(f"[ERROR] 在 {args.input_dir} 下未找到任何已知子目录")
        config.input_dirs = mapped
    if args.pdf_dir:
        config.input_dirs["pdf_download"] = args.pdf_dir
        config.input_dirs["pdf_photo"] = args.pdf_dir
    if args.word_dir:
        config.input_dirs["word"] = args.word_dir
    if args.html_dir:
        config.input_dirs["html_source"] = args.html_dir
    if args.txt_dir:
        config.input_dirs["txt"] = args.txt_dir
    if args.files:
        config.files = [f.strip() for f in args.files.split(",") if f.strip()]

    # ---- 输出 ----
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.output_prefix:
        config.output_prefix = args.output_prefix
    if args.output_types:
        types = [t.strip() for t in args.output_types.split(",") if t.strip()]
        invalid = [t for t in types if t not in VALID_OUTPUT_TYPES]
        if invalid:
            raise SystemExit(f"[ERROR] 未知输出类型: {invalid}，可选 {list(VALID_OUTPUT_TYPES)}")
        config.output_types = types

    # ---- 清洗 ----
    if args.min_text_length is not None:
        config.min_text_length = args.min_text_length
    if args.max_text_length is not None:
        config.max_text_length = args.max_text_length
    if args.dedup_threshold is not None:
        config.dedup_threshold = args.dedup_threshold
    if args.chunk_size is not None:
        config.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        config.chunk_overlap = args.chunk_overlap
    if args.max_samples is not None:
        for key in config.max_samples:
            config.max_samples[key] = args.max_samples

    # ---- LLM ----
    if args.llm:
        config.enable_llm = True
    if args.llm_provider:
        config.llm_provider = args.llm_provider
    if args.llm_model:
        if config.llm_provider == "deepseek":
            config.deepseek_model = args.llm_model
        else:
            config.dashscope_model = args.llm_model

    # ---- 质量评分 ----
    if args.no_quality:
        config.quality_enabled = False
    if args.quality_min_per_dim is not None:
        config.quality_min_per_dimension = args.quality_min_per_dim
    if args.quality_min_total is not None:
        config.quality_min_total = args.quality_min_total

    # ---- 运行控制 ----
    if args.resume:
        config.resume = True
    if args.no_cache:
        config.use_read_cache = False
    if args.no_rejected:
        config.save_rejected = False
    if args.max_workers is not None:
        config.max_workers = max(1, args.max_workers)

    # ---- API key：先加载项目根 .env（可选，真实环境变量优先），再从环境变量补齐 ----
    from .dotenv import load_dotenv
    load_dotenv()
    if not config.dashscope_api_key:
        config.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not config.deepseek_api_key:
        config.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")

    return config


def print_config_summary(config: ProcessorConfig):
    existing = config.resolved_input_dirs()
    print("\n" + "=" * 60)
    print("[Config] 工作流配置")
    print("=" * 60)
    print(f"  输入目录({len(existing)}/{len(config.input_dirs)} 存在):")
    for key, path in config.input_dirs.items():
        mark = "OK " if os.path.isdir(path) else "缺失"
        print(f"    [{mark}] {key}: {path}")
    print(f"  输出目录: {config.output_dir}")
    print(f"  输出类型: {config.output_types}")
    print(f"  模式: {'resume(断点续跑)' if config.resume else 'rebuild(重建输出)'}")
    print(f"  读取缓存: {'ON' if config.use_read_cache else 'OFF'}")
    print(f"  质量评分: {'ON' if (config.quality_enabled and config.api_key()) else 'OFF'} "
          f"(阈值 单项≥{config.quality_min_per_dimension}, 总分≥{config.quality_min_total})")
    print(f"  LLM生成: {'ON (' + config.llm_provider + ' -> ' + config.llm_model() + ')' if config.enable_llm else 'OFF (template only)'}")
    print(f"  OCR(视觉): {'ON (dashscope ' + config.ocr_model + ')' if config.ocr_api_key() else 'OFF (无 DASHSCOPE_API_KEY)'}")
    print(f"  并发数: {config.max_workers}")
    print(f"  最小/最大文本长度: {config.min_text_length} / {config.max_text_length}")
    print(f"  分块: {config.chunk_size} (overlap {config.chunk_overlap})")
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    setup_logging(args.verbose)
    config = build_config(args)

    print_config_summary(config)

    if args.clear_cache:
        cache_path = os.path.join(config.cache_dir, "read_cache.jsonl")
        ReadCache(cache_path).clear()
        print(f"[Cache] 已清空: {cache_path}")
        return

    if args.dry_run:
        pipeline = FileProcessorPipeline(config)
        files = pipeline._scan_files()
        print(f"[DryRun] 待处理文件: {len(files)} 个")
        for file_type, count in pipeline.stats.files_by_type.most_common():
            print(f"  - {file_type}: {count}")
        formatter = JSONLFormatter(config.output_dir, config.output_prefix)
        for name in config.output_types:
            print(f"  输出 {name}: {formatter.get_path(name)}")
        return

    pipeline = FileProcessorPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()