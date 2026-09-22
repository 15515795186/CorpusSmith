#!/usr/bin/env python
"""
项目进度查看工具
显示：输入数据量 / 读取缓存命中情况 / 各数据集产出 / 质量报告 / 最近日志
"""

import json
import os
import re
import sys
from collections import Counter

# Windows 控制台/重定向统一 UTF-8，避免中文输出乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = "data"
DATASETS_DIR = "datasets"
CACHE_FILE = os.path.join(".cache", "read_cache.jsonl")


def _latest_log() -> str:
    """自动选最新的 OCR 运行日志（ocr_run.log / ocr_run2.log / ...），
    也可用命令行第一个参数显式指定"""
    if len(sys.argv) > 1:
        return sys.argv[1]
    import glob as _glob
    logs = [p for p in _glob.glob("ocr_run*.log") if os.path.isfile(p)]
    return max(logs, key=os.path.getmtime) if logs else "ocr_run.log"


LOG_FILE = _latest_log()

# 需要 OCR 才能处理的目录
OCR_DIRS = ("版式pdf", "版式pdf中文版", "截图html")

# 已知的输入数据目录 → 期望扩展名（只统计这些，避免把工具链/暂存目录算进来）
KNOWN_DIRS = {
    "文本pdf": (".pdf",),
    "文本pdf中文版": (".pdf",),
    "版式pdf": (".pdf",),
    "版式pdf中文版": (".pdf",),
    "word": (".docx",),
    "源码html": (".html", ".htm", ".xhtml"),
    "截图html": (".png", ".jpg", ".jpeg", ".bmp", ".tiff"),
    "txt": (".txt",),
}


def count_input_files():
    """统计各输入目录的文件数（只统计已知数据目录 + 期望扩展名）"""
    print("=" * 56)
    print("输入数据")
    print("=" * 56)
    total = 0
    if not os.path.isdir(DATA_DIR):
        print(f"  [WARN] 未找到 {DATA_DIR}/")
        return 0

    for name, exts in KNOWN_DIRS.items():
        path = os.path.join(DATA_DIR, name)
        if not os.path.isdir(path):
            continue
        files = [f for f in os.listdir(path)
                 if os.path.isfile(os.path.join(path, f))
                 and not f.startswith(".")
                 and f.lower().endswith(exts)]
        if not files:
            continue
        total += len(files)
        tag = " (需OCR)" if name in OCR_DIRS else ""
        print(f"  {name}: {len(files)} 个文件{tag}")
    print(f"  合计: {total} 个可处理文件")
    return total


def show_datasets():
    print("\n" + "=" * 56)
    print("数据集产出")
    print("=" * 56)
    if not os.path.isdir(DATASETS_DIR):
        print(f"  [WARN] 未找到 {DATASETS_DIR}/")
        return

    found = False
    for name in sorted(os.listdir(DATASETS_DIR)):
        nested = os.path.join(DATASETS_DIR, name, "data.jsonl")
        flat = os.path.join(DATASETS_DIR, f"{name}.jsonl")
        for path in (nested, flat):
            if not os.path.exists(path):
                continue
            found = True
            sources = Counter()
            with open(path, "r", encoding="utf-8") as f:
                lines = [line for line in f if line.strip()]
            for line in lines:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                meta = rec.get("meta") or {}
                src = meta.get("source") or rec.get("source")
                if src:
                    sources[os.path.basename(str(src))] += 1
            print(f"  {path}: {len(lines)} 条记录, {len(sources)} 个来源文件")

    if not found:
        print("  (暂无输出)")


def show_quality_reports():
    if not os.path.isdir(DATASETS_DIR):
        return
    reports = [f for f in os.listdir(DATASETS_DIR) if f.startswith("quality_report")]
    if not reports:
        return
    print("\n" + "=" * 56)
    print("质量评分")
    print("=" * 56)
    for name in sorted(reports):
        path = os.path.join(DATASETS_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                report = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        print(f"  {name}: {report.get('passed')}/{report.get('total_samples')} 通过 "
              f"({report.get('pass_rate_percent')}%), API错误 {report.get('api_errors', 0)}")


def show_rejections():
    path = os.path.join(DATASETS_DIR, "rejected.jsonl")
    if not os.path.exists(path):
        return
    reasons = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                reasons[json.loads(line).get("reason", "unknown")] += 1
            except json.JSONDecodeError:
                continue
    print("\n" + "=" * 56)
    print(f"被丢弃文本 ({path})")
    print("=" * 56)
    for reason, count in reasons.most_common(10):
        print(f"  {reason}: {count}")


def show_cache():
    print("\n" + "=" * 56)
    print("读取缓存")
    print("=" * 56)
    if not os.path.exists(CACHE_FILE):
        print("  (暂无缓存)")
        return
    size_mb = os.path.getsize(CACHE_FILE) / 1024 / 1024
    count = 0
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    print(f"  {CACHE_FILE}: {count} 条已缓存读取结果 ({size_mb:.1f} MB)")
    print("  重跑时命中缓存的文件不会再次解析/调用 OCR API")


def show_run_report():
    path = os.path.join(DATASETS_DIR, "run_report.json")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    print("\n" + "=" * 56)
    print("上次运行")
    print("=" * 56)
    print(f"  时间: {report.get('started_at')} → {report.get('finished_at')} "
          f"({report.get('elapsed_seconds')}s)")
    print(f"  文件: 成功 {report.get('files_processed')} / 失败 {report.get('files_failed')}"
          f" | 缓存: {report.get('cache')}")
    print(f"  清洗后文本: {report.get('texts_cleaned')} | 丢弃: {report.get('texts_filtered')}")
    print(f"  数据集: {report.get('datasets')}")


def show_log():
    if not os.path.exists(LOG_FILE):
        print(f"\n  [WARN] 日志文件不存在 ({LOG_FILE})")
        return
    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    ocr_pages = [line for line in lines if "OCR page" in line]
    errors = [line for line in lines if "[ERROR]" in line or "[FAIL]" in line]
    print("\n" + "=" * 56)
    print(f"日志 {LOG_FILE}（共 {len(lines)} 行）")
    print("=" * 56)
    print(f"  OCR 页面进度行: {len(ocr_pages)} | 错误行: {len(errors)}")
    if ocr_pages:
        for line in ocr_pages[-3:]:
            match = re.search(r'OCR page (\d+)/(\d+)', line)
            if match:
                print(f"    最近: page {match.group(1)}/{match.group(2)}")
    for line in errors[-3:]:
        print(f"    {line.strip()[:120]}")


def main():
    print("=" * 56)
    print("项目进度查看器")
    print("=" * 56)
    count_input_files()
    show_datasets()
    show_quality_reports()
    show_rejections()
    show_cache()
    show_run_report()
    show_log()


if __name__ == "__main__":
    main()