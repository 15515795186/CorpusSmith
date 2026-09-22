# -*- coding: utf-8 -*-
"""一次性重建：input 语义修复后，从读取缓存重新转换 instruction/cot。

依据：质检门槛评的是清洗后文本（与转换逻辑无关），通过来源集合不变；
转换是纯本地确定性过程 → 零 API、不重建 pretrain。
输出顺序与原管线一致（A 批目录在前、B 批在后，各自按路径排序），
保持与 pretrain 数据的来源对应关系稳定。"""
import json
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from file_processor.cleaners.text_cleaner import TextCleaner
from file_processor.cleaners.html_cleaner import HTMLCleaner
from file_processor.converters.instruction_converter import InstructionConverter
from file_processor.converters.cot_converter import COTConverter
from file_processor.formatters.jsonl_formatter import JSONLFormatter
from file_processor.readers.base import ReaderResult

A_DIRS = ["data/文本pdf", "data/文本pdf中文版", "data/word", "data/源码html", "data/txt"]
B_DIRS = ["data/版式pdf", "data/版式pdf中文版", "data/截图html"]

# 通过质检的来源集合（来自 pretrain 记录的 meta.source）
passed = set()
with open("datasets/pretrain/data.jsonl", encoding="utf-8") as f:
    for line in f:
        passed.add(os.path.basename(str(json.loads(line)["meta"]["source"])))
print(f"通过质检来源: {len(passed)}")

cache = {}
with open(".cache/read_cache.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("path"):
            cache[os.path.normpath(rec["path"])] = rec

tc, hc = TextCleaner(), HTMLCleaner()
ic, cc = InstructionConverter(), COTConverter()
fmt = JSONLFormatter("datasets", overwrite=True)
fmt.prepare("instruction")
fmt.prepare("cot")

n_files = 0
for group in (A_DIRS, B_DIRS):
    roots = [os.path.normpath(os.path.abspath(d)) for d in group]
    entries = sorted((e for p, e in cache.items()
                      if any(p.startswith(r) for r in roots)),
                     key=lambda e: os.path.normpath(e["path"]))
    for e in entries:
        path = os.path.normpath(e["path"])
        if os.path.basename(path) not in passed:
            continue
        raw = e.get("text", "")
        if e.get("file_type") in ("html_source", "html_screenshot"):
            text = tc.clean(hc.clean(raw))
        else:
            text = tc.clean(raw)
        if not text:
            continue
        r = ReaderResult(text=text, filepath=path,
                         file_type=e.get("file_type", ""), metadata={}, success=True)
        fmt.write_batch("instruction", ic.convert(r))
        fmt.write_batch("cot", cc.convert(r))
        n_files += 1

for name in ("instruction", "cot"):
    total = fmt.count_lines(name)
    fmt.write_metadata(
        name,
        sources=len(fmt.read_sources(name)),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        config={"enable_llm": False, "rebuilt_input_grounding": True},
    )
    print(f"{name}: {total} 条")
print(f"重转换文件数: {n_files} | 指纹去重跳过: {fmt.duplicates_skipped}")
