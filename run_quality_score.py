#!/usr/bin/env python
"""
质量评分独立运行脚本（对已有数据集重新打分）

用法:
  # 默认：读取数据集，把通过质检的子集写到 data.scored.jsonl（不覆盖原文件）
  python run_quality_score.py --dataset pretrain

  # key 来源优先级：--api-key > 环境变量 > 项目根 .env

  # 覆盖原文件（会先备份为 data.jsonl.bak）
  python run_quality_score.py --dataset pretrain --in-place

  # 只抽样评分，快速估算通过率
  python run_quality_score.py --dataset instruction --sample 100 --workers 8
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows 控制台/重定向统一 UTF-8，避免中文输出乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from file_processor.quality_scorer import QualityScorer
from file_processor.dotenv import load_dotenv

TEXT_FIELDS = ("text", "output", "instruction")


def resolve_data_file(dataset: str) -> str:
    """布局 <output_dir>/<name>/data.jsonl；兼容旧的扁平 <output_dir>/<name>.jsonl"""
    nested = os.path.join("datasets", dataset, "data.jsonl")
    flat = os.path.join("datasets", f"{dataset}.jsonl")
    for path in (nested, flat):
        if os.path.exists(path):
            return path
    raise SystemExit(f"[ERROR] 未找到数据集文件: {nested}（或 {flat}）")


def load_records(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def record_text(record: dict) -> str:
    for field in TEXT_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def parse_args():
    parser = argparse.ArgumentParser(description="对已有数据集做 LLM 质量评分")
    parser.add_argument("--dataset", default="pretrain",
                        choices=["pretrain", "instruction", "cot"],
                        help="数据集名称 (default: pretrain)")
    parser.add_argument("--api-key", default=None,
                        help="DashScope API key（默认读取环境变量 DASHSCOPE_API_KEY）")
    parser.add_argument("--model", default="qwen-plus", help="评分模型")
    parser.add_argument("--min-per-dim", type=int, default=3, help="单项最低分")
    parser.add_argument("--min-total", type=int, default=20, help="总分最低分")
    parser.add_argument("--workers", type=int, default=4, help="并发数")
    parser.add_argument("--sample", type=int, default=0,
                        help="只评分前 N 条 (0=全部)")
    parser.add_argument("--in-place", action="store_true",
                        help="覆盖原文件（自动备份 .bak）")
    parser.add_argument("--output", default=None, help="输出文件路径")
    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()  # 项目根 .env 可用时自动补齐 key（真实环境变量优先）

    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "[ERROR] 未提供 API key。请设置环境变量 DASHSCOPE_API_KEY "
            "或使用 --api-key（不要把 key 写进代码）")

    data_file = resolve_data_file(args.dataset)
    records = load_records(data_file)
    if args.sample > 0:
        records = records[:args.sample]
    if not records:
        raise SystemExit(f"[ERROR] {data_file} 中没有数据")

    print(f"评分数据集: {data_file}")
    print(f"记录数: {len(records)} | 模型: {args.model} | 并发: {args.workers}")

    scorer = QualityScorer(
        api_key=api_key, model=args.model,
        min_per_dimension=args.min_per_dim, min_total=args.min_total,
        max_workers=args.workers,
    )

    texts = [record_text(r) for r in records]
    sources = [os.path.basename(str((r.get("meta") or {}).get("source")
                                     or r.get("source") or f"record_{i}"))
               for i, r in enumerate(records)]

    start = time.time()
    scored = scorer.score_batch(texts, sources, progress_callback=None)
    elapsed = time.time() - start

    passed_records = [records[i] for i, (_, score, _) in enumerate(scored) if score.passed]
    failed = len(scored) - len(passed_records)

    # 输出（默认不破坏原文件）
    if args.in_place:
        backup = data_file + ".bak"
        with open(backup, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_path = data_file
        print(f"已备份原文件: {backup}")
    else:
        output_path = args.output or os.path.join(os.path.dirname(data_file), "data.scored.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for record in passed_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report_path = os.path.join("datasets", f"quality_report_{args.dataset}.json")
    report = scorer.generate_report(scored, report_path)

    print("\n" + "=" * 52)
    print(f"评分完成: {len(scored)} 条, 通过 {len(passed_records)}, 淘汰 {failed}, "
          f"耗时 {elapsed:.0f}s ({len(scored) / max(elapsed, 1e-9):.1f} 条/秒)")
    print(f"通过率: {report['pass_rate_percent']}%")
    print(f"输出:   {output_path}")
    print(f"报告:   {report_path}")
    if report.get("api_errors"):
        print(f"[WARN] {report['api_errors']} 条因 API 错误被淘汰，请检查 key/配额")
    print("=" * 52)


if __name__ == "__main__":
    main()