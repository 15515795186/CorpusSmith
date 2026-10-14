# CorpusSmith

[![CI](https://github.com/15515795186/CorpusSmith/actions/workflows/ci.yml/badge.svg)](https://github.com/15515795186/CorpusSmith/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/15515795186/CorpusSmith)](https://github.com/15515795186/CorpusSmith/releases/latest)
[![License](https://img.shields.io/github/license/15515795186/CorpusSmith)](https://github.com/15515795186/CorpusSmith/blob/main/LICENSE)

**Forge multi-format documents into LLM training datasets: parse / OCR → clean & dedup → PPL + LLM quality scoring → emit pretrain / SFT / COT.**

**English** | [简体中文](README.md)

Reads PDF / Word / HTML / TXT / PNG in bulk and converts them into three kinds of LLM training
datasets: pretraining (`pretrain`), instruction tuning (`instruction`, SFT), and Chain-of-Thought (`cot`).

---

## Key features

| Feature | Description |
|---------|-------------|
| Incremental reuse | Read results (including OCR output) are cached on disk — re-runs never re-parse or re-call paid APIs |
| Resumable runs | `--resume` skips already-produced and deterministically rejected sources **before reading**; sources dropped due to rate limiting are retried automatically |
| Reproducible output | Files are processed in path order; every run rebuilds outputs instead of stacking records across runs |
| Diagnosable | Rejected text is written to `rejected.jsonl` with a reason histogram, making threshold and OCR-quality issues traceable |
| Deterministic reports | Each run emits `run_report.json` (file counts, cache hits, rejection reasons, dataset sizes, redacted config) |
| Credential safety | Project-root `.env` is auto-loaded (gitignored); real environment variables take precedence; keys never reach code, repo, or registry |
| Dependency-free testing | 215+ unit and end-to-end tests (stdlib `unittest`), no API key required |

---

## Quick start

**Requirements**: Python 3.10+ (developed on 3.12). Dependencies are listed in `requirements.txt`.

```bash
# 1) Install dependencies
python -m pip install -r requirements.txt

# 2) Configure credentials (optional — only LLM generation / quality scoring / OCR need them)
cp .env.example .env            # Windows PowerShell: Copy-Item .env.example .env
# Fill in DEEPSEEK_API_KEY (text chain) and DASHSCOPE_API_KEY (vision/OCR chain)

# 3) Run (defaults: read data/爬取语料, write to datasets/)
python -m file_processor.main              # full pipeline
python -m file_processor.main --no-quality # structural chain only, no credentials needed
python -m file_processor.main --dry-run    # inspect config and pending file count only
```

> **Corpus data is not committed.** Bring your own corpus for a first run, or rebuild it
> following `data/README.md`.

---

## Directory layout

```
CorpusSmith/
├── file_processor/                  # core package (34 modules, ~6,000 lines)
│   ├── main.py                      # CLI entry point
│   ├── pipeline.py                  # orchestration: scan → read → clean → score → convert → emit
│   ├── config.py                    # global config (dataclass + JSON overrides)
│   ├── dotenv.py                    # zero-dependency .env loader
│   ├── quality_scorer.py            # multi-dimension LLM quality scoring (concurrent)
│   ├── readers/                     # pdf / word / html / txt / qwen-vl ocr
│   ├── cleaners/                    # text_cleaner / html_cleaner / quality_filter / ppl_filter
│   ├── converters/                  # pretrain / instruction / cot
│   ├── formatters/                  # JSONL output layout, rebuild / resume, metadata
│   ├── llm/                         # optional dashscope generator
│   └── utils/                       # lang_detect / dedup / stats / cache
├── data/                            # source corpus + data build toolchain (not committed)
├── datasets/                        # current output
├── tests/                           # unit and end-to-end tests
├── check_progress.py                # progress: input / output / cache / report / log
├── verify_datasets.py               # output self-check
├── rebuild_derived.py               # rebuild instruction/cot from read cache at zero API cost
├── run_quality_score.py             # re-score an existing dataset under a separate scheme
├── BENCHMARK.md                     # measured data and methodology
├── CHANGELOG.md
└── README.en.md
```

---

## Pipeline

```
Raw corpus files
      │
      ▼
┌────────────────  Read (disk-cached)  ────────────────┐
│ text PDF → pymupdf/pypdf    scanned PDF → 300 DPI → Qwen-VL OCR │
│ PNG screenshot → Qwen-VL OCR    DOCX → python-docx    HTML → BeautifulSoup │
└──────────────────────────┬───────────────────────────┘
                           ▼
      Clean (control chars / whitespace / line breaks / mojibake / punctuation)
                           ▼
      Heuristic filtering (length / info density / mojibake / blank-form circuit breaker / SimHash dedup)
                           ▼
      LLM quality scoring (deepseek-flash, concurrent; per-dim ≥ 3 and total ≥ 18)
                           ▼
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   pretrain            instruction           cot
 (chunk + meta)      (template / LLM)   (reasoning chain)
```

---

## Measured results

Two separate measurement scopes exist and must not be conflated. Full methodology, per-batch
tables and caveats are in **[BENCHMARK.md](BENCHMARK.md)**.

### Pipeline validation batch — 400 files, LLM scoring enabled

| Dataset | Records | Sources | Size |
|---------|---------|---------|------|
| pretrain | 7,384 | 277 | 10.99 MB |
| instruction | 2,455 | 242 | 3.57 MB |
| cot | 175 | 148 | 0.37 MB |

- Read / OCR: 150 / 150 success on the scanned-and-screenshot half (≈1,580 PDF pages + 50 images)
- Quality-filter pass rate evolution: 8.1% → 61.3% → **69.6%**
- Dedup calibration: 250 files → 248, i.e. only 2 genuine template duplicates removed
- Self-check clean: zero exact duplicates, zero missing fields, zero U+FFFD, consistent chunk indices

### Current dataset output — `datasets/` from 24 source files

| Dataset | Records | Sources | Elapsed |
|---------|---------|---------|---------|
| pretrain | 237 | 18 | 215.2 s |
| instruction | 165 | 18 | 1,367.8 s |
| cot | 51 | 51 | 125.0 s |

---

## Output formats

```
datasets/
├── pretrain/{data.jsonl, metadata.json}
├── instruction/{data.jsonl, metadata.json}
├── cot/{data.jsonl, metadata.json}
├── quality_report.json     # per-dimension means, pass rate, top-10 failure reasons
├── run_report.json         # run statistics, redacted config snapshot
└── rejected.jsonl          # dropped text: source / stage / reason / excerpt
```

```jsonc
// pretrain
{"text": "chunked text…", "meta": {"source": "xxx.pdf", "type": "pdf_download",
  "language": "zh", "chunk_index": 0, "chunk_total": 17,
  "size_bytes": 1496, "token_count_est": 332}}

// instruction (SFT) — open-book: material must go into `input`
{"instruction": "Summarize the key points of the following material about data processing.",
 "input": "Data processing typically involves several steps…", "output": "1. …\n2. …", "source": "article.txt"}

// instruction — closed-book: only self-contained prompts may leave `input` empty
{"instruction": "What is an act of unfair competition?", "input": "",
 "output": "Unfair competition refers to…", "source": "law.txt"}

// cot — open-book enumeration
{"instruction": "How many items are listed in the material? Verify each one.",
 "input": "1. …\n2. …\n3. …", "chain_of_thought": "Counting the material item by item: …",
 "output": "The material lists 3 items.", "source": "doc.pdf"}
```

---

## Quality scoring

Cleaned text is scored by `deepseek-flash` (concurrent, controlled by `--max-workers`):

| Dimension | Range | Meaning |
|-----------|-------|---------|
| Content completeness | 1–5 | Has a proper beginning and end; not truncated |
| Language correctness | 1–5 | No mojibake, no OCR errors, grammatical |
| Information density | 1–5 | Substantive rather than vague or repetitive |
| Structural clarity | 1–5 | Clear paragraph hierarchy |
| Overall readability | 1–5 | Fluent and logically coherent |

**Threshold**: per-dimension ≥ 3 and total ≥ 20/25 by default (both adjustable). The shipped run
configs use **18**; see BENCHMARK.md for the offline-replay rationale.

**Long-document sampling**: texts over roughly 3,700 characters are sampled from head / middle /
tail (1,200 characters each), joined with an explicit omission marker that the prompt is told not
to treat as truncation.

---

## OCR

Scanned PDFs and PNG screenshots both go through **Qwen-VL-Plus** (DashScope):

- Scanned PDFs are rendered at 300 DPI and recognized page by page (`ocr_dpi` configurable);
  paragraph and table structure is preserved.
- 8-way concurrency (`max_workers`), roughly 10–20 seconds per page.
- Results enter the read cache, so `--resume` never re-bills.
- Without an API key, scanned PDFs fail loudly instead of silently falling back to a local model.

---

## CLI reference

| Flag | Description |
|------|-------------|
| `--config PATH` | JSON config file (explicit CLI flags win) |
| `--input-dir DIR` | Root directory; subdirs are auto-mapped by document type |
| `--files a,b,c` | Process only the named files |
| `--output-dir/-o` | Output directory (default `datasets`) |
| `--output-types` | Subset of `pretrain,instruction,cot` |
| `--chunk-size/--chunk-overlap` | Pretraining chunking (`0` = no chunking) |
| `--min-text-length` | Minimum text length (default 30) |
| `--dedup-threshold` | SimHash document dedup threshold (default 0.90) |
| `--max-samples` | Per-dataset sample cap (0 = unlimited) |
| `--resume` | Keep existing output; skip already-produced and deterministically rejected sources before reading |
| `--no-cache` / `--clear-cache` | Disable / clear the read cache |
| `--no-quality` | Skip LLM scoring even when a key is present |
| `--no-rejected` | Do not emit `rejected.jsonl` |
| `--max-workers` | Concurrency for reading and scoring (default 4) |
| `--dry-run` | Print config and pending file count only |
| `--llm --llm-provider dashscope` | Enable LLM-based instruction / COT generation |
| `-v/--verbose` | DEBUG logging |

---

## Testing

```bash
python -m unittest discover -s tests -v
```

The suite covers config parsing and key redaction, the `.env` loader, output layout and
rebuild/resume semantics, sample-level exact dedup, blank-form circuit breaking, read-cache
invalidation, CJK punctuation preservation, dedup fingerprint stability against golden samples,
quality-scoring sampling strategy and response parsing, every reader (text/scanned PDF routing,
OCR temp cleanup, GBK encoding, HTML type detection, inline-link paragraph reassembly), all three
converters, and pipeline end-to-end runs (idempotent re-runs, resume skipping, metadata source
semantics). **No API key is required**, and no file outside the workspace is written.

---

## Design notes

- **CJK punctuation**: normalization uses NFC rather than NFKC (NFKC folds `，` into `,`);
  full-width → half-width conversion applies to English text only.
- **Long documents are not discarded**: `max_text_length` defaults to 1,000,000, so 60 KB+ papers
  go through chunking instead of being dropped for length.
- **Mojibake detection by ratio**: the old "single character appearing >100 times" rule was
  replaced with consecutive-repeat plus single-character-ratio checks, because frequent CJK
  characters were being killed.
- **Quality-scoring input must be self-consistent**: the old implementation sent `text[:3000]`
  while the prompt penalized mid-sentence truncation — a systematic self-rejection that pushed
  the pass rate down to 8.1%. Head/middle/tail sampling with an omission marker restored it.
- **Dedup fingerprinting**: `blake2b` (64-bit) replaces the built-in `hash()`, whose per-process
  salt made dedup results drift between runs (12 vs 8 documents removed, only 4 in common).
- **Token granularity**: CJK adjacent-character and English adjacent-word bigrams, because
  single characters / bare words over-kill on same-domain corpora.
- **`input` field semantics**: samples whose answer comes from a specific document must place the
  material in `input`; leaving it empty for such samples teaches the model to hallucinate.
  `verify_datasets.py` cannot catch this (it checks key presence, not semantics) — 18 converter
  contract tests lock it down instead.
- **Output rebuild and resume**: each run clears the target `data.jsonl` by default; `--resume`
  appends and moves the skip decision ahead of reading. Deterministically rejected sources
  (clean/filter stage) are skipped too, since SimHash depends on same-run context — re-running a
  near-duplicate alone would otherwise let it slip through and pollute the dataset.

---

## License

Released under the [MIT License](LICENSE).

Copyright (c) 2026 Qmm
