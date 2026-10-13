"""扫码模式: 对 data/爬取语料/ 全部分类语料算 PPL, 输出分布供阈值标定。

覆盖范围(共 80 份):
  - .txt / .docx / .html / 有文字层的 .pdf: 直接提取
  - 纯图 PDF(版式pdf*) 与 截图 PNG: 先走 Qwen-VL OCR 取文本再打分
    OCR 结果缓存到 ocr_cache/, 重跑复用, 不重复消耗 API

用法:
  python data/_pipeline/scan_ppl_distribution.py
  可选: --no-ocr    只跑有文字层的语料(原行为, 70 份)
        --only-ocr  只跑需要 OCR 的语料(30 份, 走缓存)
环境要求:
  HUGGINGFACE_HUB_CACHE=<PPL 模型缓存根>(可选; 未设置则用 HF 默认缓存)
  DASHSCOPE_API_KEY(OCR 用, 自动从项目 .env 读取)

输出:
  - 终端: 总体直方图 + 分位数表 + 每类语料分位摘要 + 拟淘汰清单
  - 文件: data/_pipeline/ppl_scan_scores.json(每文件 PPL + 文本来源 text/ocr)
  - 缓存: data/_pipeline/ocr_cache/<分类>__<文件名>.txt(OCR 文本)

分位口径: 全文统一用 ceil(n*q)-1(第 ceil(n*q) 个样本为界, 1 起算),
          与 file_processor.cleaners.ppl_filter.PPLFilter.calibrate_threshold 完全一致,
          避免"分位表"与"建议阈值"各说各话。
"""
import json
import math
import os
import sys
import time
from collections import defaultdict

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PKG_ROOT)

# PPL 模型缓存根: 优先 PPL_CACHE_DIR / HUGGINGFACE_HUB_CACHE 环境变量; 均未设置则用 HF 默认缓存
if os.environ.get("PPL_CACHE_DIR"):
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["PPL_CACHE_DIR"])

from file_processor.cleaners.ppl_filter import PPLFilter  # noqa: E402
from file_processor.cleaners.text_cleaner import TextCleaner  # noqa: E402
from file_processor.cleaners.html_cleaner import HTMLCleaner  # noqa: E402
from file_processor.config import ProcessorConfig  # noqa: E402
from file_processor.dotenv import load_dotenv  # noqa: E402
from file_processor.readers.pdf_reader import PDFReader  # noqa: E402
from file_processor.readers.ocr_reader import QwenVLOcrReader  # noqa: E402

BASE = os.path.join(_PKG_ROOT, "data", "爬取语料")
_PIPE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_PIPE, "ppl_scan_scores.json")
OCR_CACHE = os.path.join(_PIPE, "ocr_cache")

TEXT_EXTS = (".txt", ".docx", ".html", ".pdf")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def quantile_index(n: int, q: float) -> int:
    """统一分位取法(与 PPLFilter.calibrate_threshold 一致): 第 ceil(n*q) 个样本(1 起算)"""
    return min(max(int(math.ceil(n * q)) - 1, 0), n - 1)


def quantiles(sorted_scores: list, qs=(0.5, 0.75, 0.9, 0.95, 0.99)):
    n = len(sorted_scores)
    out = {}
    for q in qs:
        out[q] = sorted_scores[quantile_index(n, q)] if n else None
    return out


# ---------------- 无 OCR 的文本提取 ----------------
def read_text_plain(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        raw = open(path, "rb").read()
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            import chardet
            enc = chardet.detect(raw)["encoding"] or "gb18030"
            return raw.decode(enc, errors="replace")
    if ext == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    if ext == ".html":
        return open(path, encoding="utf-8", errors="replace").read()
    if ext == ".pdf":
        import fitz
        doc = fitz.open(path)
        try:
            return "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()
    return ""


# ---------------- OCR 结果缓存 ----------------
def cache_path(category: str, name: str) -> str:
    return os.path.join(OCR_CACHE, f"{category}__{name}.txt")


def cache_load(category: str, name: str):
    p = cache_path(category, name)
    if os.path.isfile(p):
        return open(p, encoding="utf-8").read()
    return None


def cache_save(category: str, name: str, text: str):
    os.makedirs(OCR_CACHE, exist_ok=True)
    with open(cache_path(category, name), "w", encoding="utf-8") as f:
        f.write(text)


# ---------------- 主流程 ----------------
def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    no_ocr = "--no-ocr" in argv
    only_ocr = "--only-ocr" in argv

    if not os.path.isdir(BASE):
        print("未找到语料目录:", BASE)
        return 1

    files = []
    for root, dirs, names in os.walk(BASE):
        # 跳过下划线开头的辅助目录(如 _src 里是源 PDF 原件, 不是语料)
        dirs[:] = [d for d in dirs if not d.startswith("_")]
        for name in names:
            if name.startswith("_manifest") or name == "manifest":
                continue
            if os.path.splitext(name)[1].lower() in TEXT_EXTS + IMAGE_EXTS:
                files.append(os.path.join(root, name))
    files.sort()

    load_dotenv()
    cfg = ProcessorConfig()
    api_key = cfg.api_key()
    ocr_reader = QwenVLOcrReader(
        api_key=api_key, model=cfg.ocr_model, max_retries=cfg.ocr_max_retries)
    pdf_reader = PDFReader(
        min_text_chars=cfg.ocr_min_text_chars,
        qwen_ocr=ocr_reader if api_key else None,
        ocr_dpi=cfg.ocr_dpi,
        work_dir=os.path.join(_PKG_ROOT, cfg.ocr_work_dir),
    )

    if not api_key:
        print("!! 未找到 DASHSCOPE_API_KEY: 纯图 PDF / 截图 PNG 将被跳过\n")

    print(f"扫描 {len(files)} 份文件(no_ocr={no_ocr}, only_ocr={only_ocr}) ...")
    cleaner = TextCleaner()
    html_cleaner = HTMLCleaner()
    flt = PPLFilter(threshold=None, quantile=0.90)

    records = []
    ocr_calls = 0
    for i, p in enumerate(files, 1):
        cat = os.path.basename(os.path.dirname(p))
        name = os.path.basename(p)
        ext = os.path.splitext(p)[1].lower()
        tag = f"[{i}/{len(files)}] {cat}/{name}"
        need_ocr = ext in IMAGE_EXTS
        text, source = "", "text"

        try:
            # 1) 图片: 必须 OCR(优先用缓存)
            if need_ocr:
                if no_ocr or not api_key:
                    records.append({"category": cat, "file": name, "ppl": None,
                                    "skipped": "no_ocr_available"})
                    print(f"{tag}: 跳过(需 OCR, 当前未启用)")
                    continue
                text = cache_load(cat, name)
                if text is None:
                    t0 = time.time()
                    r = ocr_reader.read(p)
                    ocr_calls += 1
                    if not r.success:
                        records.append({"category": cat, "file": name, "ppl": None,
                                        "skipped": f"ocr_error:{r.error}"})
                        print(f"{tag}: OCR 失败 {r.error}")
                        continue
                    text = r.text
                    cache_save(cat, name, text)
                    print(f"{tag}: OCR 完成 {len(text)} 字符 ({time.time()-t0:.0f}s, 已缓存)")
                else:
                    print(f"{tag}: OCR 缓存命中 {len(text)} 字符")
                source = "ocr"

            # 2) PDF: 让 PDFReader 判定文字层/扫描件(纯图自动转 OCR 并缓存)
            elif ext == ".pdf" and not only_ocr:
                cached = cache_load(cat, name)
                if cached is not None:
                    text, source = cached, "ocr"
                    print(f"{tag}: OCR 缓存命中(PDF) {len(text)} 字符")
                else:
                    r = pdf_reader.read(p)
                    if not r.success:
                        records.append({"category": cat, "file": name, "ppl": None,
                                        "skipped": f"read_error:{r.error}"})
                        print(f"{tag}: 读取失败 {r.error}")
                        continue
                    text = r.text
                    if r.metadata.get("used_ocr"):
                        source = "ocr"
                        ocr_calls += 1
                        cache_save(cat, name, text)
                        print(f"{tag}: 纯图 PDF -> OCR 完成 {len(text)} 字符 (已缓存)")

            # 3) 其余文本类: 直接提取; only_ocr 模式下跳过
            else:
                if only_ocr:
                    records.append({"category": cat, "file": name, "ppl": None,
                                    "skipped": "not_ocr_target"})
                    continue
                text = read_text_plain(p)

            # 统一清洗(必过): txt/docx/文字层 PDF/OCR 文本 一律走 TextCleaner, html 先过 HTMLCleaner。
            # 踩坑: 早前版本只在 source=="ocr" 时清洗, 导致文字层 PDF 拿"未清洗原文"打分,
            # 与首次扫码(旧脚本)及流水线口径都不可比 —— 所有分支必须同一清洗口径。
            text = cleaner.clean(html_cleaner.clean(text) if ext == ".html" else text) or ""

        except Exception as exc:  # noqa: BLE001
            records.append({"category": cat, "file": name, "ppl": None,
                            "skipped": f"error:{exc}"})
            print(f"{tag}: 失败 {exc}")
            continue

        if len(text.strip()) < 20:
            records.append({"category": cat, "file": name, "ppl": None,
                            "skipped": "too_short", "source": source})
            print(f"{tag}: 跳过(过短, 清洗后 {len(text.strip())} 字符, 来源 {source})")
            continue

        try:
            r = flt.check(text)
        except Exception as exc:  # noqa: BLE001
            records.append({"category": cat, "file": name, "ppl": None,
                            "skipped": f"error:{exc}", "source": source})
            print(f"{tag}: 打分失败 {exc}")
            continue
        records.append({"category": cat, "file": name, "ppl": round(r.ppl, 2),
                        "source": source})
        if i % 5 == 0 or i == len(files):
            n_ok = sum(1 for x in records if x["ppl"] is not None)
            print(f"[{i}/{len(files)}] 已扫描, n={n_ok}")

    json.dump(records, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n分数已存: {OUT}(OCR 调用 {ocr_calls} 次, 缓存目录 {OCR_CACHE})")

    scored = [r["ppl"] for r in records if r["ppl"] is not None]
    if not scored:
        print("无有效分数")
        return 1
    ss = sorted(scored)
    q = quantiles(ss)
    print("\n===== 总体 PPL 分布(n=%d) =====" % len(ss))
    print("  min=%.2f  p50=%.2f  p75=%.2f  p90=%.2f  p95=%.2f  p99=%.2f  max=%.2f"
          % (ss[0], q[0.5], q[0.75], q[0.9], q[0.95], q[0.99], ss[-1]))
    n_ocr = sum(1 for r in records if r.get("source") == "ocr" and r["ppl"] is not None)
    print("  其中 OCR 来源 %d 份, 文字层来源 %d 份" % (n_ocr, len(ss) - n_ocr))

    # ASCII 直方图(20 桶)
    lo, hi = ss[0], ss[-1]
    bins = [0] * 20
    width = max((hi - lo) / 20, 1e-9)
    for s in ss:
        bins[min(int((s - lo) / width), 19)] += 1
    print("\n  直方图(每格=最高%.1f%%样本):" % (100.0 * max(bins) / len(ss)))
    for b, c in enumerate(bins):
        print("  %8.1f | %s %d" % (lo + b * width, "#" * int(60 * c / max(bins)), c))

    # 每类语料分位摘要(90/95 淘汰参考)
    print("\n===== 每类语料分位(vs 全局 90/95) =====")
    by_cat = defaultdict(list)
    for r in records:
        if r["ppl"] is not None:
            by_cat[r["category"]].append(r["ppl"])
    for cat in sorted(by_cat):
        cs = sorted(by_cat[cat])
        cq = quantiles(cs)
        print("  %-14s n=%-3d p50=%.1f p90=%.1f p95=%.1f max=%.1f"
              % (cat, len(cs), cq[0.5], cq[0.9], cq[0.95], cs[-1]))

    # 未纳入名单(必须显式列出, 避免"覆盖率静默缺失")
    skipped = [r for r in records if r["ppl"] is None]
    if skipped:
        print("\n===== 未纳入打分 %d 份 =====" % len(skipped))
        for r in skipped:
            print("      %-14s %s  (%s)" % (r["category"], r["file"], r.get("skipped")))

    # 拟淘汰清单(q90 / q95) —— 阈值与下方建议值同源(ceil 口径)
    print("\n===== 拟淘汰清单 =====")
    for qkey in (0.9, 0.95):
        th = q[qkey]
        victims = [r for r in records if r["ppl"] is not None and r["ppl"] > th]
        print("  [q%d=%.2f] 淘汰 %d 份(%.1f%%):" % (qkey * 100, th, len(victims),
                                                    100.0 * len(victims) / len(ss)))
        for r in sorted(victims, key=lambda x: -x["ppl"])[:12]:
            print("      %.2f  %s/%s" % (r["ppl"], r["category"], r["file"]))

    # 标定到 config 的建议值
    flt.scores = ss
    for qk in (0.9, 0.95):
        t = flt.calibrate_threshold(qk)
        same = "一致" if abs(t - q[qk]) < 1e-9 else "!! 与分位表不一致"
        print("\n[建议] ppl_quantile=%.0f%% -> ppl_threshold=%.2f (%s)"
              % (qk * 100, t, same))
    return 0


if __name__ == "__main__":
    sys.exit(main())