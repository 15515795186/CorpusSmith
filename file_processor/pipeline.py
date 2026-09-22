"""
核心管道编排器
文件发现 → 读取(带缓存) → 清洗 → 质量评分 → 转换 → 输出

设计要点：
- 确定性：文件按路径排序后处理，输出顺序稳定、可复现
- 可重跑：每次运行默认重建输出；--resume 时保留已有输出并跳过已处理来源
- 可诊断：被清洗/过滤/质检淘汰的文本写入 rejected.jsonl，便于定位阈值问题
- 省成本：读取结果（含 OCR）落盘缓存，重跑不重复调用 API
"""

import glob
import json
import logging
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional

from .config import ProcessorConfig
from .readers.base import ReaderResult
from .readers.pdf_reader import PDFReader
from .readers.word_reader import WordReader
from .readers.html_reader import HTMLReader
from .readers.txt_reader import TxtReader
from .readers.ocr_reader import QwenVLOcrReader
from .cleaners.text_cleaner import TextCleaner
from .cleaners.html_cleaner import HTMLCleaner
from .cleaners.quality_filter import QualityFilter
from .converters.pretrain_converter import PretrainConverter
from .converters.instruction_converter import InstructionConverter
from .converters.cot_converter import COTConverter
from .formatters.jsonl_formatter import JSONLFormatter
from .utils.cache import ReadCache
from .utils.stats import StatsCollector
from .utils.lang_detect import detect_language, estimate_token_count
from .quality_scorer import QualityScorer

logger = logging.getLogger(__name__)

# 输入目录 key → 扩展名
EXTENSIONS_MAP = {
    "pdf_download": (".pdf",),
    "pdf_download_cn": (".pdf",),
    "pdf_photo": (".pdf",),
    "pdf_photo_cn": (".pdf",),
    "word": (".docx",),
    "html_source": (".html", ".htm", ".xhtml"),
    "html_screenshot": (".html", ".htm"),
    "screenshot": (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"),
    "txt": (".txt",),
}


class FileProcessorPipeline:
    """文件处理主管道"""

    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.stats = StatsCollector()
        self.started_at = datetime.now()
        self._t0 = time.time()
        self.rejections: Counter = Counter()
        self.rejected_path = os.path.join(config.output_dir, "rejected.jsonl")
        self._rejected_file = None

        # Qwen-VL OCR 读取器（截图 + 扫描PDF）
        self.qwen_ocr = QwenVLOcrReader(
            api_key=config.api_key(),
            model=config.ocr_model,
            max_retries=config.ocr_max_retries,
        )

        # 读取器（按顺序匹配 can_handle）
        self.readers = [
            PDFReader(
                min_text_chars=config.ocr_min_text_chars,
                qwen_ocr=self.qwen_ocr if config.api_key() else None,
                ocr_dpi=config.ocr_dpi,
                work_dir=config.ocr_work_dir,
            ),
            WordReader(),
            HTMLReader(),
            TxtReader(),
            self.qwen_ocr,
        ]

        # 读取缓存
        self.cache = ReadCache(
            path=os.path.join(config.cache_dir, "read_cache.jsonl"),
            enabled=config.use_read_cache,
        )

        # 清洗器
        self.text_cleaner = TextCleaner()
        self.html_cleaner = HTMLCleaner()
        self.quality_filter = QualityFilter(
            min_length=config.min_text_length,
            max_length=config.max_text_length,
            garbage_ratio=config.garbage_ratio_threshold,
            target_languages=config.target_languages,
            dedup_threshold=config.dedup_threshold,
            keep_unknown_language=config.keep_unknown_language,
            max_repeat_char_ratio=config.max_repeat_char_ratio,
        )

        # LLM生成器（可选）
        self.llm_generator = self._init_llm_generator()

        # 转换器
        self.pretrain_converter = PretrainConverter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_samples=config.max_samples.get("pretrain", 0),
        )
        self.instruction_converter = InstructionConverter(
            max_samples=config.max_samples.get("instruction", 0),
            llm_generator=self.llm_generator if config.enable_llm else None,
        )
        self.cot_converter = COTConverter(
            max_samples=config.max_samples.get("cot", 0),
            llm_generator=self.llm_generator if config.enable_llm else None,
        )

        # 输出（resume 时不截断，否则每次运行重建，避免重复样本堆积）
        self.formatter = JSONLFormatter(
            output_dir=config.output_dir,
            prefix=config.output_prefix,
            overwrite=not config.resume,
        )

        # 质量评分器（启用且存在 API key 时生效）
        self.quality_scorer = None
        if config.quality_enabled and config.api_key():
            self.quality_scorer = QualityScorer(
                api_key=config.api_key(),
                model=config.quality_model,
                min_per_dimension=config.quality_min_per_dimension,
                min_total=config.quality_min_total,
                max_workers=config.max_workers,
            )
            logger.info("质量评分已启用 (%s, 阈值: 单项≥%d, 总分≥%d)",
                        config.quality_model, config.quality_min_per_dimension,
                        config.quality_min_total)
        elif config.quality_enabled:
            logger.warning("未找到 DashScope API key，质量评分已跳过")

    def _init_llm_generator(self):
        """初始化LLM生成器"""
        if not self.config.enable_llm:
            return None

        if self.config.llm_provider == "openai":
            from .llm.openai_generator import OpenAIGenerator
            return OpenAIGenerator(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url,
                model=self.config.openai_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
        if self.config.llm_provider == "dashscope":
            from .llm.dashscope_generator import DashScopeGenerator
            return DashScopeGenerator(
                api_key=self.config.api_key(),
                model=self.config.dashscope_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
        logger.warning("Unknown LLM provider: %s", self.config.llm_provider)
        return None

    # ================= 主流程 =================
    def run(self):
        """运行完整pipeline"""
        from tqdm import tqdm

        logger.info("=" * 60)
        logger.info("[Pipeline] 文件处理管道启动")
        logger.info("=" * 60)

        # 1. 扫描文件（排序保证可复现）
        files = self._scan_files()
        if not files:
            logger.warning("[WARN] 未找到任何可处理的文件")
            return

        self.stats.total_files = len(files)
        logger.info("[Files] 发现 %d 个文件", len(files))

        # 2. 准备输出目录（非 resume 时清空旧结果）
        if not self.config.resume:
            self.formatter.prepare_all(self.config.output_types)
        else:
            logger.info("[Output] resume 模式：保留已有输出，跳过已处理来源")

        try:
            # 3. 读取
            pbar = tqdm(total=len(files), desc="[Read] 读取文件", unit="file",
                        ncols=70, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
            results = self._read_all(files, pbar)
            pbar.close()

            # 4. 清洗 + 启发式过滤
            pbar2 = tqdm(total=len(results), desc="[Clean] 清洗数据", unit="file",
                         ncols=70, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
            cleaned_results = self._clean_all(results, pbar2)
            pbar2.close()

            # 5. 质量评分 + 过滤（启用时）
            if self.quality_scorer is not None:
                cleaned_results = self._quality_check(cleaned_results)

            # 6. 转换并输出
            converters = {
                "pretrain": self.pretrain_converter,
                "instruction": self.instruction_converter,
                "cot": self.cot_converter,
            }
            for name in ("pretrain", "instruction", "cot"):
                if name in self.config.output_types:
                    self._convert_and_output(name, cleaned_results, converters[name])

            # 7. 统计与运行报告
            self.stats.print_summary()
            self._write_run_report(files, cleaned_results)
        finally:
            self._close_rejected()

    # ================= 文件扫描 =================
    def _scan_files(self) -> List[str]:
        """扫描所有输入目录，收集文件（按路径排序，保证输出确定性）"""
        if self.config.files:
            result = [f for f in self.config.files if os.path.isfile(f)]
            missing = [f for f in self.config.files if not os.path.isfile(f)]
            for f in missing:
                logger.warning("[WARN] 指定的文件不存在: %s", f)
            return sorted(result)

        all_files: List[str] = []
        seen_paths = set()

        for file_type, dir_path in self.config.input_dirs.items():
            if not dir_path:
                continue
            if not os.path.isdir(dir_path):
                logger.warning("[WARN] 输入目录不存在，已跳过: %s (%s)", dir_path, file_type)
                continue

            exts = EXTENSIONS_MAP.get(file_type, ())
            if not exts:
                logger.warning("[WARN] 未知的输入类型: %s", file_type)
                continue

            for ext in exts:
                for pattern in (os.path.join(dir_path, f"*{ext}"),
                                os.path.join(dir_path, "**", f"*{ext}")):
                    for fp in glob.glob(pattern, recursive=True):
                        norm = os.path.normpath(os.path.abspath(fp))
                        if norm in seen_paths:
                            continue
                        seen_paths.add(norm)
                        all_files.append(fp)
                        self.stats.files_by_type[file_type] += 1

        return sorted(all_files)

    # ================= 读取 =================
    def _read_all(self, files: List[str], pbar=None) -> List[ReaderResult]:
        """并发读取所有文件（带磁盘缓存）"""
        results: List[ReaderResult] = []
        max_workers = min(self.config.max_workers, len(files))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._read_single, fp): fp for fp in files
            }
            for future in as_completed(future_to_file):
                fp = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result.success:
                        self.stats.files_processed += 1
                    else:
                        self.stats.files_failed += 1
                        if result.error:
                            self.stats.add_error(os.path.basename(fp), result.error)
                except Exception as e:
                    self.stats.files_failed += 1
                    self.stats.add_error(os.path.basename(fp), str(e))
                if pbar:
                    pbar.update(1)

        results.sort(key=lambda r: r.filepath)  # 与扫描顺序一致
        return results

    def _read_single(self, filepath: str) -> ReaderResult:
        """读取单个文件（优先命中缓存）"""
        cached = self.cache.get(filepath)
        if cached is not None:
            self.stats.cache_hits += 1
            return ReaderResult(
                text=cached.get("text", ""),
                filepath=filepath,
                file_type=cached.get("file_type", "unknown"),
                metadata=dict(cached.get("metadata") or {}),
                success=True,
            )

        for reader in self.readers:
            if reader.can_handle(filepath):
                result = reader.read(filepath)
                break
        else:
            result = ReaderResult(
                text="", filepath=filepath, file_type="unknown",
                success=False, error=f"No reader for: {filepath}"
            )

        # 只缓存成功且有内容的读取结果（失败下次重试）
        if result.success and result.text:
            self.cache.put(filepath, result.text, result.file_type, result.metadata)
        return result

    # ================= 清洗 =================
    def _clean_all(self, results: List[ReaderResult], pbar=None) -> List[ReaderResult]:
        """清洗 + 启发式质量过滤"""
        cleaned: List[ReaderResult] = []

        for result in results:
            if not result.success or not result.text:
                if pbar:
                    pbar.update(1)
                continue

            source = os.path.basename(result.filepath)
            try:
                if result.file_type in ("html_source", "html_screenshot"):
                    clean = self.html_cleaner.clean(result.text)
                else:
                    clean = self.text_cleaner.clean(result.text)

                if not clean:
                    self._record_rejection(source, "clean", "empty_after_clean", result.text)
                    self.stats.texts_filtered += 1
                    if pbar:
                        pbar.update(1)
                    continue

                quality = self.quality_filter.check(clean)
                if not quality.passed:
                    self._record_rejection(source, "filter", quality.reason or "unknown", clean)
                    self.stats.texts_filtered += 1
                    if pbar:
                        pbar.update(1)
                    continue

                result.text = clean

                lang = detect_language(clean)
                result.metadata["language"] = lang
                self.stats.language_dist[lang] += 1
                self.stats.total_tokens_est += estimate_token_count(clean, lang)

                cleaned.append(result)
                self.stats.texts_cleaned += 1
            except Exception as e:
                logger.warning("清洗失败: %s - %s", source, e)
                self._record_rejection(source, "clean", f"exception: {e}", result.text)
            if pbar:
                pbar.update(1)

        logger.info("[Clean] 清洗完成: %d -> %d (过滤 %d)",
                    len(results), len(cleaned), self.stats.texts_filtered)
        if self.rejections:
            top = ", ".join(f"{r}={c}" for r, c in self.rejections.most_common(5))
            logger.info("[Clean] 丢弃原因 TOP5: %s", top)
        return cleaned

    # ================= 质量评分 =================
    def _quality_check(self, results: List[ReaderResult]) -> List[ReaderResult]:
        """LLM 质量评分，只保留通过质检的结果（并发执行）"""
        from tqdm import tqdm
        if not results:
            return []

        texts = [r.text for r in results]
        sources = [os.path.basename(r.filepath) for r in results]

        pbar = tqdm(total=len(texts), desc="[Quality] 质量评分", unit="text",
                    ncols=70, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")

        def on_done(done, total, score):
            pbar.update(1)

        scored = self.quality_scorer.score_batch(texts, sources, progress_callback=on_done)
        pbar.close()

        passed_results = []
        for i, (text, score, source) in enumerate(scored):
            if score.passed:
                passed_results.append(results[i])
            else:
                logger.info("  淘汰: %s (总分 %d/25) - %s",
                            source, score.total_score,
                            "; ".join(score.issues[:2]) or score.error or "")
                self._record_rejection(source, "quality", score.error or "low_score", text)

        report = self.quality_scorer.generate_report(scored, self.formatter.get_report_path())

        self.stats.texts_filtered += report["failed"]
        if report.get("api_errors"):
            logger.warning("[Quality] 有 %d 条因 API 错误被淘汰，请检查 key/配额",
                           report["api_errors"])

        logger.info("[Quality] 质量评分: %d -> %d (淘汰 %d, 通过率 %.1f%%)",
                    len(results), len(passed_results),
                    report["failed"], report["pass_rate_percent"])
        logger.info("[Quality] 报告已保存: %s", self.formatter.get_report_path())

        return passed_results

    # ================= 转换输出 =================
    def _convert_and_output(self, name: str, results: List[ReaderResult], converter):
        """转换并输出数据集"""
        from tqdm import tqdm

        existing_sources = self.formatter.read_sources(name) if self.config.resume else set()
        if existing_sources:
            logger.info("[Output] [%s] resume: 已有 %d 个来源，跳过", name, len(existing_sources))

        total_samples = 0
        total_generated = 0
        written_sources = 0

        pbar = tqdm(total=len(results), desc=f"[Output] {name}", unit="file",
                    ncols=70, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")

        for result in results:
            try:
                source = os.path.basename(result.filepath)
                if source in existing_sources:
                    pbar.update(1)
                    continue
                samples = converter.convert(result)
                if samples:
                    written = self.formatter.write_batch(name, samples)
                    total_generated += len(samples)
                    total_samples += written
                    if written:
                        written_sources += 1
            except Exception as e:
                logger.error("转换失败 [%s] %s: %s",
                             name, os.path.basename(result.filepath), e)
            pbar.update(1)
        pbar.close()

        if total_generated != total_samples:
            logger.info("[Output] [%s] 跳过与数据集中已有记录完全相同的样本 %d 条",
                        name, total_generated - total_samples)

        if name == "pretrain":
            self.stats.pretrain_samples = total_samples
        elif name == "instruction":
            self.stats.instruction_samples = total_samples
        elif name == "cot":
            self.stats.cot_samples = total_samples

        output_path = self.formatter.get_path(name)
        self.formatter.write_metadata(
            name,
            sources=written_sources,
            generated_at=datetime.now().isoformat(timespec="seconds"),
            elapsed_seconds=round(time.time() - self._t0, 1),
            config={
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "enable_llm": self.config.enable_llm,
                "quality_enabled": self.quality_scorer is not None,
                "resume": self.config.resume,
            },
        )
        logger.info("[Output] [%s] %d records (%d sources) -> %s",
                    name, total_samples, written_sources, output_path)

    # ================= 报告 =================
    def _record_rejection(self, source: str, stage: str, reason: str, text: str = ""):
        """记录被丢弃的文本，便于事后诊断阈值/OCR质量"""
        self.rejections[reason] += 1
        if not self.config.save_rejected:
            return
        if self._rejected_file is None:
            os.makedirs(os.path.dirname(os.path.abspath(self.rejected_path)) or ".", exist_ok=True)
            self._rejected_file = open(self.rejected_path, "w", encoding="utf-8")
        record = {
            "source": source,
            "stage": stage,
            "reason": reason,
            "chars": len(text or ""),
            "text": (text or "")[:self.config.rejected_max_chars],
        }
        self._rejected_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._rejected_file.flush()

    def _close_rejected(self):
        if self._rejected_file is not None:
            self._rejected_file.close()
            self._rejected_file = None

    def _write_run_report(self, files: List[str], cleaned_results: List[ReaderResult]):
        report = {
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.time() - self._t0, 1),
            "total_files": len(files),
            "files_processed": self.stats.files_processed,
            "files_failed": self.stats.files_failed,
            "cache": self.cache.summary(),
            "texts_cleaned": self.stats.texts_cleaned,
            "texts_filtered": self.stats.texts_filtered,
            "rejection_reasons": dict(self.rejections.most_common()),
            "exact_duplicate_samples_skipped": self.formatter.duplicates_skipped,
            "language_dist": dict(self.stats.language_dist.most_common()),
            "total_tokens_est": self.stats.total_tokens_est,
            "datasets": {
                name: self.formatter.count_lines(name)
                for name in self.config.output_types
            },
            "config": self.config.to_dict(redact=True),
        }
        path = os.path.join(self.config.output_dir, "run_report.json")
        os.makedirs(self.config.output_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("[Report] 运行报告已保存: %s", path)