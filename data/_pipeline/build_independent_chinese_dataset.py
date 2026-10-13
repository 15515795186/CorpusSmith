#!/usr/bin/env python3
"""Build four independent Chinese datasets from public external sources."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

import requests
from bs4 import BeautifulSoup
from docx import Document
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "_pipeline"
STAGE = PIPELINE / "_external_chinese_stage"

OUTPUT_DIRS = {
    "screenshot_html": ROOT / "截图html",
    "source_html": ROOT / "源码html",
    "word_docx": ROOT / "word",
    "txt": ROOT / "txt",
}
GENERATED_MARKER = ".generated_by_build_mixed_dataset.json"

GOV_LIST_URL = "https://www.gov.cn/yaowen/liebiao/YAOWENLIEBIAO.json"
WORD_CANDIDATES = PIPELINE / "gov_edu_word_candidates.json"
WIKISOURCE_API = "https://zh.wikisource.org/w/api.php"

def _browser_candidates() -> list[Path]:
    """浏览器可执行文件候选：优先 CHROME_PATH 环境变量，其次 PATH，最后常见安装位置。"""
    candidates: list[Path] = []
    env_override = os.environ.get("CHROME_PATH")
    if env_override:
        candidates.append(Path(env_override))
    for name in ("chrome", "msedge", "chromium", "chromium-browser", "google-chrome"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    candidates += [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    return candidates


CHROME_CANDIDATES = _browser_candidates()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/140.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cjk_count(text: str) -> int:
    return sum("\u4e00" <= char <= "\u9fff" for char in text)


def normalized_text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    return sha256_bytes(normalized.encode("utf-8"))


def parse_selectable_text(raw: bytes) -> str | None:
    for encoding in ("utf-8-sig", "utf-16", "utf-8", "gb18030", "cp1252"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\x00" in text:
            continue
        if text.count("\ufffd") / max(len(text), 1) > 0.001:
            continue
        return text
    return None


def safe_filename(value: str, fallback: str, limit: int = 64) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", "_", value).strip("._")
    return (value[:limit].rstrip("._") or fallback)


def requests_get(url: str, timeout: int = 25) -> requests.Response:
    response = requests.get(
        url,
        headers=HEADERS,
        timeout=timeout,
        allow_redirects=True,
    )
    response.raise_for_status()
    return response


def ensure_empty_outputs(force: bool) -> None:
    conflicts: list[Path] = []
    for directory in OUTPUT_DIRS.values():
        directory.mkdir(parents=True, exist_ok=True)
        conflicts.extend(path for path in directory.iterdir() if path.is_file())
    if conflicts and not force:
        preview = "\n".join(str(path) for path in conflicts[:10])
        raise RuntimeError(
            "目标数据目录已有文件。使用 --force 仅替换采集器记录过的文件。\n"
            f"{preview}"
        )


def reset_stage() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    for key in OUTPUT_DIRS:
        (STAGE / key).mkdir(parents=True, exist_ok=True)


def output_relative(category: str, filename: str) -> str:
    return str((OUTPUT_DIRS[category] / filename).relative_to(ROOT))


def find_chrome() -> Path:
    for candidate in CHROME_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise RuntimeError("Chrome/Edge executable not found")


def capture_rendered_page(chrome: Path, source_url: str, target: Path) -> bytes:
    with tempfile.TemporaryDirectory(prefix="external_chinese_chrome_") as profile:
        command = [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--disable-software-rasterizer",
            "--use-gl=swiftshader",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-background-networking",
            "--disable-extensions",
            "--hide-scrollbars",
            "--no-first-run",
            "--no-default-browser-check",
            "--force-device-scale-factor=1",
            "--window-size=1440,1200",
            "--virtual-time-budget=5000",
            f"--user-data-dir={profile}",
            f"--screenshot={target}",
            "--dump-dom",
            source_url,
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
        )
        for _ in range(30):
            if target.is_file() and target.stat().st_size > 0:
                break
            time.sleep(0.1)
        if target.is_file() and target.stat().st_size > 0 and len(result.stdout) >= 5000:
            return result.stdout.encode("utf-8")
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"Chrome did not render {source_url}; "
            f"returncode={result.returncode}; {detail[-500:]}"
        )


def validate_png(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        grayscale = image.convert("L")
        histogram = grayscale.histogram()
        non_white = sum(histogram[:245])
        extrema = grayscale.getextrema()
    if width < 1000 or height < 700:
        raise RuntimeError(f"unexpected screenshot dimensions: {path} {width}x{height}")
    if non_white < 2000 or extrema[0] == extrema[1]:
        raise RuntimeError(f"screenshot appears blank: {path}")
    return {
        "width": width,
        "height": height,
        "non_white_pixels": non_white,
        "grayscale_min": extrema[0],
        "grayscale_max": extrema[1],
    }


def collect_html_and_screenshots() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from replace_html_with_complex_wikisource import collect_pairs

    return collect_pairs(STAGE)


def docx_title(path: Path, fallback: str) -> tuple[str, int]:
    document = Document(path)
    title = (document.core_properties.title or "").strip()
    if not title:
        for paragraph in document.paragraphs:
            text = " ".join(paragraph.text.split())
            if 2 <= len(text) <= 80:
                title = text
                break
    title = title or fallback
    body_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    return title, cjk_count(body_text)


def docx_xml_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    return re.sub(r"<[^>]+>", " ", xml)


def select_docx_candidates(limit: int = 50) -> list[dict[str, Any]]:
    candidates = json.loads(WORD_CANDIDATES.read_text(encoding="utf-8"))
    unique_hashes: set[str] = set()
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in sorted(
        candidates,
        key=lambda item: (
            -int(item["cjk_chars"]),
            str(item.get("source_url") or item.get("url") or ""),
        ),
    ):
        path = Path(candidate["file"])
        if not path.is_file():
            continue
        digest = str(candidate.get("content_sha256") or sha256_file(path))
        if digest in unique_hashes:
            continue
        xml_text = docx_xml_text(path).casefold()
        if any(term in xml_text for term in ("简历", "求职", "应聘", "resume")):
            continue
        unique_hashes.add(digest)
        candidate = dict(candidate)
        candidate["sha256"] = digest
        source_url = str(candidate.get("source_url") or candidate.get("url") or "")
        category = urlparse(source_url).netloc or "root"
        by_category[category].append(candidate)

    selected: list[dict[str, Any]] = []
    category_order = sorted(
        by_category,
        key=lambda category: (-len(by_category[category]), category),
    )
    while len(selected) < limit:
        advanced = False
        for category in category_order:
            if by_category[category]:
                selected.append(by_category[category].pop(0))
                advanced = True
                if len(selected) >= limit:
                    break
        if not advanced:
            break
    if len(selected) < limit:
        raise RuntimeError(f"only {len(selected)} unique validated DOCX candidates")
    return selected


def collect_docx() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for candidate in select_docx_candidates(50):
        source = Path(candidate["file"])
        source_url = str(candidate.get("source_url") or candidate.get("url") or "")
        try:
            title, _ = docx_title(
                source,
                str(candidate.get("link_text") or Path(urlparse(source_url).path).stem),
            )
        except Exception:
            title = str(candidate.get("link_text") or Path(urlparse(source_url).path).stem)
        title = re.sub(r"(?i)\.docx$", "", title).strip()
        body_cjk = int(candidate["cjk_chars"])
        if body_cjk < 100:
            continue
        index = len(items) + 1
        sample_id = (
            f"{index:03d}_gov_word_"
            f"{safe_filename(urlparse(source_url).netloc, 'source')}_"
            f"{safe_filename(title, 'document')}"
        )
        target = STAGE / "word_docx" / f"{sample_id}.docx"
        shutil.copyfile(source, target)
        with zipfile.ZipFile(target) as archive:
            if "word/document.xml" not in archive.namelist():
                target.unlink(missing_ok=True)
                continue
        items.append(
            {
                "sample_id": sample_id,
                "language": "zh-CN",
                "title": title,
                "source_family": urlparse(source_url).netloc,
                "source_url": source_url,
                "page_url": candidate.get("page_url", source_url),
                "transform": "direct public DOCX attachment downloaded unchanged",
                "file": output_relative("word_docx", target.name),
                "bytes": target.stat().st_size,
                "sha256": candidate["sha256"],
                "cjk_chars": body_cjk,
            }
        )
        print(f"docx={len(items)} {source_url}", flush=True)
    if len(items) < 50:
        raise RuntimeError(f"only {len(items)} usable DOCX files")
    return items


def wikisource_titles(category: str = "古文", limit: int = 500) -> list[str]:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmnamespace": "0",
        "cmlimit": str(limit),
    }
    response = requests.get(
        WIKISOURCE_API,
        params=params,
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    return [
        item["title"]
        for item in response.json().get("query", {}).get("categorymembers", [])
        if item.get("ns") == 0 and item.get("title")
    ]


def wikisource_revisions(titles: list[str]) -> list[dict[str, Any]]:
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "titles": "|".join(titles),
        "prop": "revisions|info",
        "rvprop": "ids|content",
        "rvslots": "main",
        "inprop": "url",
    }
    for attempt in range(5):
        response = requests.get(
            WIKISOURCE_API,
            params=params,
            headers=HEADERS,
            timeout=45,
        )
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else 20
            time.sleep(delay * (attempt + 1))
            continue
        response.raise_for_status()
        return response.json().get("query", {}).get("pages", [])
    raise RuntimeError("Wikisource rate limit persisted after retries")


def wikitext_to_text(content: str) -> str:
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    while "{{" in text and "}}" in text:
        reduced = re.sub(r"\{\{[^{}]*\}\}", "", text)
        if reduced == text:
            break
        text = reduced
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^[*#:;]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^==+\s*(.*?)\s*==+$", r"\1", text, flags=re.MULTILINE)
    text = text.replace("'''", "").replace("''", "")
    text = html.unescape(text).replace("\xa0", " ")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def collect_txt() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    seen_text_hashes: set[str] = set()
    titles = wikisource_titles("中华人民共和国政府白皮书", 500)
    for start in range(0, min(len(titles), 300), 20):
        batch = titles[start : start + 20]
        for page in wikisource_revisions(batch):
            if len(items) >= 50:
                break
            title = str(page.get("title", "")).strip()
            revisions = page.get("revisions") or []
            if not revisions:
                continue
            content = (
                revisions[0]
                .get("slots", {})
                .get("main", {})
                .get("content", "")
            )
            text = wikitext_to_text(str(content))
            if not title or title in seen_titles or len(text) < 500:
                continue
            chinese_chars = cjk_count(text)
            if chinese_chars < 500:
                continue
            text_hash = normalized_text_hash(text)
            if text_hash in seen_text_hashes:
                continue
            seen_titles.add(title)
            seen_text_hashes.add(text_hash)

            index = len(items) + 1
            sample_id = f"{index:03d}_wikisource_{safe_filename(title, 'text')}"
            target = STAGE / "txt" / f"{sample_id}.txt"
            raw = (text.rstrip() + "\n").encode("utf-8")
            target.write_bytes(raw)
            items.append(
                {
                    "sample_id": sample_id,
                    "language": "zh-CN",
                    "title": title,
                    "source_family": "zh.wikisource.org",
                    "source_url": page.get("fullurl", ""),
                    "pageid": page.get("pageid"),
                    "revision": revisions[0].get("revid"),
                    "transform": "Wikisource raw revision converted to UTF-8 plain text",
                    "file": output_relative("txt", target.name),
                    "bytes": len(raw),
                    "sha256": sha256_bytes(raw),
                    "text_chars": len(text),
                    "cjk_chars": chinese_chars,
                }
            )
            print(f"txt={len(items)} {title}", flush=True)
        if len(items) >= 50:
            break
        time.sleep(2)
    if len(items) < 50:
        raise RuntimeError(f"only {len(items)} usable Wikisource TXT files")
    return items


def validate_stage(
    screenshot_items: list[dict[str, Any]],
    source_items: list[dict[str, Any]],
    docx_items: list[dict[str, Any]],
    txt_items: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = {
        "screenshot_html": 50,
        "source_html": 50,
        "word_docx": 50,
        "txt": 50,
    }
    actual = {
        "screenshot_html": len(list((STAGE / "screenshot_html").glob("*.png"))),
        "source_html": len(list((STAGE / "source_html").glob("*.html"))),
        "word_docx": len(list((STAGE / "word_docx").glob("*.docx"))),
        "txt": len(list((STAGE / "txt").glob("*.txt"))),
    }
    if actual != expected:
        raise RuntimeError(f"stage count mismatch: expected={expected}, actual={actual}")

    item_sets = {
        "screenshot_html": screenshot_items,
        "source_html": source_items,
        "word_docx": docx_items,
        "txt": txt_items,
    }
    for name, items in item_sets.items():
        hashes = [item["sha256"] for item in items]
        if len(set(hashes)) != len(hashes):
            duplicates = [digest for digest, count in Counter(hashes).items() if count > 1]
            raise RuntimeError(f"{name} contains duplicate hashes: {duplicates}")

    if {item["sample_id"] for item in screenshot_items} != {
        item["sample_id"] for item in source_items
    }:
        raise RuntimeError("screenshot and source HTML sample IDs differ")

    for item in screenshot_items:
        validate_png(STAGE / "screenshot_html" / f"{item['sample_id']}.png")

    for item in source_items:
        path = STAGE / "source_html" / f"{item['sample_id']}.html"
        soup = BeautifulSoup(path.read_bytes(), "lxml")
        text = " ".join(soup.get_text(" ", strip=True).split())
        if cjk_count(text) < 300:
            raise RuntimeError(f"source HTML Chinese validation failed: {path}")

    for item in docx_items:
        path = STAGE / "word_docx" / Path(item["file"]).name
        with zipfile.ZipFile(path) as archive:
            if "word/document.xml" not in archive.namelist():
                raise RuntimeError(f"invalid DOCX ZIP: {path}")
        text = docx_xml_text(path)
        if cjk_count(text) < 100:
            raise RuntimeError(f"DOCX Chinese validation failed: {path}")

    for item in txt_items:
        path = STAGE / "txt" / Path(item["file"]).name
        text = parse_selectable_text(path.read_bytes())
        if not text or cjk_count(text) < 500:
            raise RuntimeError(f"TXT Chinese validation failed: {path}")

    return {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "language": "zh-CN",
        "counts": actual,
        "unique_sha256": {
            name: len({item["sha256"] for item in items})
            for name, items in item_sets.items()
        },
        "minimum_cjk_chars": {
            "source_html": min(item["cjk_chars"] for item in source_items),
            "word_docx": min(item["cjk_chars"] for item in docx_items),
            "txt": min(item["cjk_chars"] for item in txt_items),
        },
        "checks": {
            "screenshot_html": "live gov.cn page, valid nonblank PNG, unique",
            "source_html": "rendered DOM from the same live page, >=300 Chinese characters",
            "word_docx": "direct public DOCX attachment, valid Word ZIP, unique",
            "txt": "public Wikisource full-text extract, UTF-8, unique",
        },
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def promote_stage(force: bool) -> None:
    for key, destination in OUTPUT_DIRS.items():
        destination.mkdir(parents=True, exist_ok=True)
        existing = [
            path
            for path in destination.iterdir()
            if path.is_file() and path.name != GENERATED_MARKER
        ]
        marker_path = destination / GENERATED_MARKER
        managed_files: set[str] = set()
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            managed_files = set(marker.get("files", []))

        unmanaged = [path for path in existing if path.name not in managed_files]
        if unmanaged:
            preview = "\n".join(str(path) for path in unmanaged[:10])
            raise RuntimeError(f"destination contains unmanaged files:\n{preview}")
        if existing and not force:
            raise RuntimeError(f"destination is not empty: {destination}")

        if force and marker_path.is_file():
            for name in managed_files:
                target = destination / name
                if target.is_file():
                    target.unlink()

        staged_files = sorted(
            source for source in (STAGE / key).iterdir() if source.is_file()
        )
        for source in staged_files:
            target = destination / source.name
            if target.exists():
                raise RuntimeError(f"refusing to overwrite: {target}")
            shutil.move(str(source), str(target))

        write_json(
            marker_path,
            {
                "dataset": "independent_chinese_200",
                "category": key,
                "files": [source.name for source in staged_files],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    if STAGE.exists():
        shutil.rmtree(STAGE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_empty_outputs(force=args.force)
    reset_stage()

    print("[1/4] collecting live Chinese HTML pages and screenshots")
    screenshot_items, source_items = collect_html_and_screenshots()

    print("[2/4] selecting real public Chinese DOCX attachments")
    docx_items = collect_docx()

    print("[3/4] collecting public Chinese full texts from Wikisource")
    txt_items = collect_txt()

    print("[4/4] validating and promoting")
    report = validate_stage(screenshot_items, source_items, docx_items, txt_items)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "language": "zh-CN",
        "source_policy": (
            "Independent external sources: live gov.cn HTML pages and screenshots, "
            "direct public DOCX attachments, and Wikisource Chinese full-text exports. "
            "No local paper PDF or PDF-derived content is used."
        ),
        "categories": {
            "screenshot_html": screenshot_items,
            "source_html": source_items,
            "word_docx": docx_items,
            "txt": txt_items,
        },
    }
    write_json(PIPELINE / "manifest_mixed.json", manifest)
    write_json(PIPELINE / "mixed_validation.json", report)
    promote_stage(force=args.force)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
