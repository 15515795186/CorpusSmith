"""重建 data/爬取语料/版式pdf英文版 的 10 份语料

背景
----
旧 10 份(scan_en_01..10)实为 Project Gutenberg 的**下载/封面页**, OCR 出来只有书名 + 下载链接
(767~1843 字符), 不是正文 —— 已作废重爬。

规则(与既有 版式pdf* 形态一致)
-----------------------------
- 每份 1 页、**无文字层**(需 OCR)。
- 取源 PDF 里第一页"正文页": 可提取文字 >= MIN_CHARS(默认 300), 跳过封面/版权/目录页;
  整篇无文字层(纯扫描件)时退化为"文字最多的页", 仍无文字则取第 3 页, 并在 manifest 标注选择方式。
- 150 dpi 渲染该页 -> PIL 合成单页图片 PDF。

来源
----
NASA / NOAA / DOE / NIST 的公开文档(美国政府作品, public domain), 按 COT 四类
(解释/建议/比较/推理 = 20/30/30/20)配比选材。

产物
----
- data/爬取语料/版式pdf英文版/scan_en_01..10_<slug>.pdf
- data/爬取语料/版式pdf英文版/_manifest_scan_en.json(来源 URL/标题/源页索引/sha256/时间)
- 源 PDF 原件留在 _src/(便于复查, 不参与流水线: 扫描时按扩展名只取 .pdf 顶层? 见下)

用法
----
  D:/Anaconda3/envs/Unlimited-OCR/python.exe data/_pipeline/rebuild_scan_en_pdf.py [--dry-run]

注: 本脚本产物落在**顶层目录**; _src/ 子目录里的源 PDF 不会被扫码/流水线当成语料
   (扫码脚本 os.walk 会递归, 故 _src 命名以下划线开头并在扫码脚本中排除)。
"""
import hashlib
import json
import os
import re
import shutil
import sys
import time

import requests

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEST_DIR = os.path.join(PKG_ROOT, "data", "爬取语料", "版式pdf英文版")
SRC_DIR = os.path.join(DEST_DIR, "_src")
MANIFEST = os.path.join(DEST_DIR, "_manifest_scan_en.json")

MIN_CHARS = 800           # 正文页门槛: 源页可提取文字
MIN_SENTENCES = 8         # 且句末标点数 >= 8(挡"标题+作者列表"/目录页)
FIRST_N_PAGES = 15        # 优先在前 N 页里找正文页(避免跑到参考文献/附录)
DPI = 150
CACHE_DIR = os.path.join(PKG_ROOT, "data", "_pipeline", "ocr_cache")
# 句末标点(中英): 用于判定"像正文"而不是"标题/目录/封面"
SENT_RE = re.compile(r"[.!?]\s|[。；：!?]")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

# (序号, slug, COT 类别, 机构, URL)
SOURCES = [
    (1, "mwee_guide",            "解释", "NOAA",
     "https://www.noaa.gov/sites/default/files/2022-09/MWEE-Guide.pdf"),
    (2, "meteorology_educator",  "解释", "NASA GPM",
     "https://gpm.nasa.gov/education/sites/default/files/lesson_plan_files/Meteorology%20-%20An%20Educators%20Resource.pdf"),
    (3, "knowledge_transitions", "建议", "NASA",
     "https://nasa.gov/wp-content/uploads/2025/10/sustaining-knowledge-through-transitions.pdf"),
    (4, "mariners_guide",        "建议", "NOAA NHC",
     "https://www.nhc.noaa.gov/pdf/Mariners_Guide_Article.pdf"),
    (5, "om_best_practices",     "建议", "DOE FEMP",
     "https://www.energy.gov/sites/default/files/2020/04/f74/omguide_complete_w-eo-disclaimer.pdf"),
    (6, "comparing_instruments", "比较", "NIST",
     "https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.2106.pdf"),
    (7, "hardness_five_methods","比较", "NIST",
     "https://nvlpubs.nist.gov/nistpubs/nbstechnologic/nbstechnologicpaperT11.pdf"),
    (8, "flight_control_math",   "推理", "NASA",
     "https://nasa.gov/sites/default/files/atoms/files/aam-flight-control-math1-graphing-educator-guide_0.pdf"),
    (9, "exploring_the_extreme", "推理", "NASA",
     "https://science.nasa.gov/wp-content/uploads/2023/10/Exploring_the_Extreme_Educators_Guide.pdf"),
    (10, "thermocouple_methods", "推理", "NIST",
     "https://nvlpubs.nist.gov/nistpubs/jres/14/jresv14n3p247_A1b.pdf"),
]

# 备用池: 任一主源失败时按序替补(同类别优先)
FALLBACK = [
    ("exploring_the_moon",  "解释", "NASA", "https://science.nasa.gov/wp-content/uploads/2024/01/exploring-the-moon-teachers-guide.pdf"),
    ("industrial_waste",    "建议", "EPA",  "https://www.epa.gov/sites/default/files/2016-03/documents/industrial-waste-guide.pdf"),
    ("natural_gas_users",   "建议", "EPA",  "https://www.epa.gov/system/files/documents/2025-01/natural-gas-and-oil-users-guide.pdf"),
    ("pest_control",        "建议", "EPA",  "https://www.epa.gov/sites/default/files/2017-08/documents/citizens_guide_to_pest_control_and_pesticide_safety.pdf"),
    ("rj_antenna_manual",   "推理", "NASA", "https://radiojove.gsfc.nasa.gov/radio_telescope/manuals/RJ_Antenna_Manual_v2.1_2025_02_07.pdf"),
]


def import_fitz():
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    return fitz


def download(url: str, dest: str, attempts: int = 5) -> dict:
    """下载并按魔法字节校验(%PDF); 失败抛异常"""
    last = ""
    for i in range(attempts):
        try:
            with requests.get(url, headers={"User-Agent": UA}, timeout=60, stream=True) as r:
                if r.status_code != 200:
                    last = f"HTTP {r.status_code}"
                else:
                    data = r.content
                    if not data.startswith(b"%PDF"):
                        last = f"非 PDF(前 8 字节 {data[:8]!r})"
                    else:
                        with open(dest, "wb") as f:
                            f.write(data)
                        return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(4 * (i + 1))
    raise RuntimeError(f"下载失败 {url}: {last}")


def is_front_matter(text: str) -> bool:
    """识别"前置页": 封面/标题页/目录/图表清单(这些不是正文)"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    head = " ".join(lines[:3]).lower()
    if "contents" in head or re.match(r"^\W*list of (tables|figures)", head):
        return True
    # "List of Tables/Figures" 可能出现在页首编号之后, 且每行都是 "Table N ..." 会被
    # 长行占比稀释 —— 故直接扫页首 200 字符(踩坑: TN2106 第 6 页 66 行, 行数占比法漏判)
    if re.search(r"list of (tables|figures)", text[:200], re.I):
        return True
    if len(re.findall(r"\.{2,}", text)) >= 3:          # 目录点线 "....."
        return True
    numbered = sum(1 for l in lines if re.match(r"^\d+[\.\s]", l))
    if numbered / len(lines) > 0.3:                    # 行首编号清单
        return True
    tf = sum(1 for l in lines if re.match(r"^(table|figure)\s+\d", l, re.I))
    if tf / len(lines) > 0.25:                         # 图表清单
        return True
    if sum(1 for l in lines if len(l) < 40) / len(lines) > 0.7:
        return True                                    # 全是短行 = 标题/作者页
    return False


def body_score(text: str) -> float:
    """近似"正文似度"(仅在顺序搜索未果时作兜底择优)"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return -1.0
    return (len(SENT_RE.findall(text)) * 3.0
            + sum(1 for l in lines if len(l) >= 60)
            - (len(re.findall(r"\.{2,}", text)) * 4.0
               + sum(1 for l in lines if len(l) < 40) * 0.5))


def pick_content_page(doc) -> tuple:
    """选"正文页": 顺序找**第一页**既不是前置页(封面/目录/图表清单)、又像正文的页。

    规则演进(踩坑记录):
      1) "第一个 >= 300 字符的页" -> 选中封面(标题+作者列表);
      2) 加 ">=1200 字符 且 >=5 个句末标点" -> 仍选中目录页("1. INTRODUCTION .. 3");
      3) 改成"全局 body_score 最高页" -> 跑到**参考文献/附录**去了(第 65/91/243 页);
      4) 最终: 顺序扫描 + is_front_matter 否决 + (字符数, 句末标点数) 双门槛。
    """
    texts = [pg.get_text() for pg in doc]
    n_pages = len(texts)
    scan_limit = min(n_pages, max(FIRST_N_PAGES, int(n_pages * 0.3)))
    for scope, tag in ((range(scan_limit), "front_section"), (range(n_pages), "whole_doc")):
        for i in scope:
            s = texts[i].strip()
            if len(s) < MIN_CHARS or len(SENT_RE.findall(s)) < MIN_SENTENCES:
                continue
            if is_front_matter(s):
                continue
            return i, f"first_content_page({tag})", len(s)
    # 兜底: 达标页里取 body_score 最高 -> 文字最多的页 -> 第 3 页
    cands = [(body_score(t), len(t.strip()), i) for i, t in enumerate(texts)
             if len(t.strip()) >= MIN_CHARS]
    if cands:
        _, n, i = max(cands)
        return i, "best_body_score_fallback", n
    cand2 = [(len(t.strip()), i) for i, t in enumerate(texts) if t.strip()]
    if cand2:
        n, i = max(cand2)
        return i, "max_text_page", n
    return min(2, n_pages - 1), "no_text_layer_fallback", 0


def render_single_page_pdf(src_pdf: str, page_idx: int, out_pdf: str) -> int:
    """把 src_pdf 的 page_idx 渲染成单页无文字层图片 PDF, 返回图页像素宽"""
    fitz = import_fitz()
    from PIL import Image
    doc = fitz.open(src_pdf)
    try:
        pix = doc[page_idx].get_pixmap(dpi=DPI)
        tmp_png = out_pdf + ".png"
        pix.save(tmp_png)
    finally:
        doc.close()
    img = Image.open(tmp_png).convert("RGB")
    w, h = img.size
    img.save(out_pdf, "PDF", resolution=DPI)
    img.close()
    os.remove(tmp_png)
    return w


def verify_no_text_layer(path: str) -> tuple:
    """确认产物是 1 页且无可提取文字(即无文字层)"""
    fitz = import_fitz()
    doc = fitz.open(path)
    try:
        pages = doc.page_count
        chars = sum(len(pg.get_text().strip()) for pg in doc)
    finally:
        doc.close()
    return pages, chars


def redo_slugs(slugs, dry: bool) -> int:
    """定向重做: 用 _src/ 里的源 PDF 重选正文页 -> 重渲染 -> 清该份 OCR 缓存 -> 更新 manifest。

    用于"内容抽检发现某份仍是封面页"时的定点重做, 不影响其他已通过的语料。
    """
    if not os.path.isfile(MANIFEST):
        print("!! manifest 不存在, 无法重做")
        return 1
    man = json.load(open(MANIFEST, encoding="utf-8"))
    fitz = import_fitz()
    done = 0
    for slug in slugs:
        rec = next((r for r in man if r["file"].startswith("scan_en_") and slug in r["file"]), None)
        if rec is None:
            print(f"!! manifest 中无 slug={slug}")
            continue
        src_path = os.path.join(SRC_DIR, f"{slug}.pdf")
        if not os.path.isfile(src_path):
            print(f"!! 缺源文件 {src_path}")
            continue
        doc = fitz.open(src_path)
        try:
            page_idx, how, chars = pick_content_page(doc)
            page_count = doc.page_count
        finally:
            doc.close()
        if dry:
            print(f"   (dry-run) {rec['file']} -> 第 {page_idx+1} 页 ({how}, {chars} 字符)")
            continue
        out_path = os.path.join(DEST_DIR, rec["file"])
        w = render_single_page_pdf(src_path, page_idx, out_path)
        pages, out_chars = verify_no_text_layer(out_path)
        assert pages == 1 and out_chars == 0, f"{rec['file']} 产物校验失败"
        cp = os.path.join(CACHE_DIR, f"版式pdf英文版__{rec['file']}.txt")
        if os.path.isfile(cp):
            os.remove(cp)
        rec.update({
            "source_page_count": page_count, "source_page_index": page_idx,
            "source_page_selection": how, "source_page_text_chars": chars,
            "output_text_chars": out_chars, "output_image_width_px": w,
            "output_bytes": os.path.getsize(out_path),
            "rendered_at": time.strftime("%Y-%m-%d %H:%M:%S"), "redo": True,
        })
        print(f"   {rec['file']}: 重选第 {page_idx+1} 页({how}, {chars} 字符) -> 已重渲染, OCR 缓存已清")
        done += 1
    if not dry:
        json.dump(man, open(MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n重做 {done}/{len(slugs)} 份, manifest 已更新")
    return 0 if (dry or done == len(slugs)) else 1


def main(argv) -> int:
    dry = "--dry-run" in argv
    if "--redo" in argv:
        slugs = [a for a in argv[argv.index("--redo") + 1:] if not a.startswith("--")]
        if not slugs:
            print("用法: --redo <slug> [<slug> ...]")
            return 1
        return redo_slugs(slugs, dry)
    os.makedirs(SRC_DIR, exist_ok=True)

    # 清理旧 OCR 缓存(旧 10 份已作废)
    cache_dir = CACHE_DIR
    removed = 0
    if os.path.isdir(cache_dir):
        for f in os.listdir(cache_dir):
            if f.startswith("版式pdf英文版__"):
                os.remove(os.path.join(cache_dir, f))
                removed += 1
    print(f"已清理旧版式pdf英文版 OCR 缓存: {removed} 份")

    # 现有文件名(重爬替换: 全部移除, 由 manifest 记录新集合)
    old = sorted(f for f in os.listdir(DEST_DIR)
                 if f.lower().endswith(".pdf") and not f.startswith("_"))
    if old:
        # 备份到 D 盘(D:/hermes/cache/scratch); 跨盘不能用 os.replace, 用 shutil.move
        backup = r"D:/hermes/cache/scratch/_old_scan_en_gutenberg"
        os.makedirs(backup, exist_ok=True)
        for f in old:
            if not dry:
                shutil.move(os.path.join(DEST_DIR, f), os.path.join(backup, f))
        print(f"旧语料 {len(old)} 份已移出项目目录 -> {backup}")

    queue = list(SOURCES)
    fallback = list(FALLBACK)
    records = []
    idx = 0
    while len(records) < len(SOURCES):
        if queue:
            seq, slug, cat, org, url = queue.pop(0)
        else:
            slug, cat, org, url = fallback.pop(0)
            seq = len(records) + 1
        name = f"scan_en_{seq:02d}_{slug}.pdf"
        src_path = os.path.join(SRC_DIR, f"{slug}.pdf")
        out_path = os.path.join(DEST_DIR, name)
        print(f"\n[{len(records)+1}/10] {cat} | {org} | {slug}")
        if dry:
            print("   (dry-run) 跳过")
            records.append({"file": name, "url": url, "dry_run": True})
            continue
        try:
            if os.path.isfile(src_path):
                info = {"bytes": os.path.getsize(src_path),
                        "sha256": hashlib.sha256(open(src_path, "rb").read()).hexdigest()}
                print(f"   源文件已存在, 复用 {info['bytes']}B")
            else:
                print(f"   下载 {url}")
                info = download(url, src_path)
                print(f"   下载完成 {info['bytes']}B sha256={info['sha256'][:12]}…")
            fitz = import_fitz()
            doc = fitz.open(src_path)
            try:
                page_idx, how, chars = pick_content_page(doc)
                page_count = doc.page_count
            finally:
                doc.close()
            w = render_single_page_pdf(src_path, page_idx, out_path)
            pages, out_chars = verify_no_text_layer(out_path)
            assert pages == 1 and out_chars == 0, f"产物校验失败 pages={pages} chars={out_chars}"
            print(f"   源共 {page_count} 页, 取第 {page_idx+1} 页({how}, 提取文字 {chars} 字符)"
                  f" -> {name} 1 页/无文字层, 宽 {w}px, {os.path.getsize(out_path)}B")
            records.append({
                "file": name, "url": url, "org": org, "cot_category": cat,
                "source_pdf_bytes": info["bytes"], "source_pdf_sha256": info["sha256"],
                "source_page_count": page_count, "source_page_index": page_idx,
                "source_page_selection": how, "source_page_text_chars": chars,
                "output_text_chars": out_chars, "output_image_width_px": w,
                "output_bytes": os.path.getsize(out_path),
                "rendered_at": time.strftime("%Y-%m-%d %H:%M:%S"), "dpi": DPI,
            })
        except Exception as exc:  # noqa: BLE001
            print(f"   !! 失败: {exc}")
            if not queue and not fallback:
                print("   主源+备用池均已用尽, 终止")
                break
            print("   换下一个候选重试")

    if not dry:
        json.dump(records, open(MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\nmanifest: {MANIFEST}({len(records)} 条)")
        from collections import Counter
        print("类别分布:", dict(Counter(r["cot_category"] for r in records)))
        print("机构分布:", dict(Counter(r["org"] for r in records)))
    return 0 if len(records) == len(SOURCES) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))