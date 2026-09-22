"""产出数据集自检：结构、去重、来源、编码

用法: python verify_datasets.py [输出目录]   （默认 datasets）
退出码: 0=全部通过, 1=发现问题
"""
import json
import os
import sys
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = sys.argv[1] if len(sys.argv) > 1 else "datasets"
OK = True


def load(name):
    path = os.path.join(OUT, name, "data.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check(name, required):
    global OK
    path = os.path.join(OUT, name, "data.jsonl")
    if not os.path.exists(path):
        print(f"[ISSUE] {name}: 未找到 {path}")
        OK = False
        return []
    records = load(name)
    meta_path = os.path.join(OUT, name, "metadata.json")
    meta = json.load(open(meta_path, encoding="utf-8"))
    exact = len(records) - len({json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records})
    sources = Counter((r.get("meta") or {}).get("source") or r.get("source") for r in records)
    missing_keys = sum(1 for r in records if any(k not in r for k in required))
    bad_fffd = sum(1 for r in records if "\ufffd" in json.dumps(r, ensure_ascii=False))
    status = "OK" if (exact == 0 and missing_keys == 0 and meta["records"] == len(records)) else "ISSUE"
    if status == "ISSUE":
        OK = False
    print(f"[{status}] {name}: records={len(records)} (metadata={meta['records']}) "
          f"sources={len(sources)} exact_dup={exact} missing_keys={missing_keys} U+FFFD={bad_fffd}")
    print(f"        size={os.path.getsize(os.path.join(OUT, name, 'data.jsonl'))/1024/1024:.2f} MB "
          f"top_source={sources.most_common(1)[0][0][:40]} ({sources.most_common(1)[0][1]} records)")
    return records


pretrain = check("pretrain", ["text", "meta"])
instruction = check("instruction", ["instruction", "input", "output", "source"])
cot = check("cot", ["instruction", "input", "chain_of_thought", "output", "source"])

# 分块一致性
bad_chunks = 0
for r in pretrain:
    meta = r["meta"]
    if "chunk_index" in meta:
        if meta["chunk_index"] >= meta["chunk_total"]:
            bad_chunks += 1
print(f"[{'OK' if bad_chunks == 0 else 'ISSUE'}] pretrain 分块索引一致性: bad={bad_chunks}")
OK = OK and bad_chunks == 0

# 被丢弃文本
rej_path = os.path.join(OUT, "rejected.jsonl")
rej = []
if os.path.exists(rej_path):
    rej = [json.loads(l) for l in open(rej_path, encoding="utf-8") if l.strip()]
    print(f"\nrejected: {len(rej)} 条")
    for reason, count in Counter(r["reason"] for r in rej).most_common():
        print(f"  {reason}: {count}")
else:
    print("\nrejected.jsonl 不存在（可能使用了 --no-rejected）")

# 抽样查看
for label, recs in (("pretrain", pretrain), ("instruction", instruction), ("cot", cot)):
    print(f"--- {label} 样本 ---")
    if recs:
        print(json.dumps(recs[0], ensure_ascii=False)[:260])
    else:
        print("  (空)")

# 判重的来源清单
print("\n--- 被丢弃的来源（前 10）---")
for r in rej[:10]:
    print(f"  {r['source'][:60]} | {r['reason']} | {r['chars']} chars")

print("\nRESULT:", "ALL OK" if OK else "HAS ISSUES")
sys.exit(0 if OK else 1)