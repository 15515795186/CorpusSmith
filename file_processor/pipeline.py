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
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List

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
from .cleaners.ppl_filter import PPLFilter
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
        # 生成阶段(来源级并发)的写盘锁: formatter 去重指纹集合非线程安全,
        # 所有"写数据集/写淘汰记录"的路径都在此锁内串行
        self._write_lock = threading.Lock()
        self.rejected_path = os.path.join(config.output_dir, "rejected.jsonl")
        # 质量评分"API/网络错误"来源单独落盘: 与"真低质淘汰"区分开 ——
        # 断网时被误淘汰的来源可按此清单定点补跑, 不必整轮重评
        self.quality_api_error_path = os.path.join(config.output_dir, "quality_api_error.jsonl")
        self.quality_exempted_path = os.path.join(config.output_dir, "quality_exempted.jsonl")
        self._exempt_file = None
        self._rejected_file = None
        self._api_error_file = None

        # Qwen-VL OCR 读取器（截图 + 扫描PDF）
        self.qwen_ocr = QwenVLOcrReader(
            api_key=config.ocr_api_key(),
            model=config.ocr_model,
            max_retries=config.ocr_max_retries,
        )

        # 读取器（按顺序匹配 can_handle）
        self.readers = [
            PDFReader(
                min_text_chars=config.ocr_min_text_chars,
                qwen_ocr=self.qwen_ocr if config.ocr_api_key() else None,
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

        # 困惑度过滤器(免费预筛闸): 默认不启用; enable_ppl_filter 时延迟加载本地小 LM
        self.ppl_filter = None
        if config.enable_ppl_filter:
            self.ppl_filter = PPLFilter(
                model_name=config.ppl_model,
                threshold=config.ppl_threshold or None,
                quantile=config.ppl_quantile,
                cache_dir=config.ppl_cache_dir or None,
            )

        # LLM生成器（可选）；COT 转换器可用 llm_cot_model 指定专用模型
        self.llm_generator = self._init_llm_generator()
        self.cot_llm_generator = self._init_cot_llm_generator(self.llm_generator)

        # 转换器
        self.pretrain_converter = PretrainConverter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            max_samples=config.max_samples.get("pretrain", 0),
        )
        self.instruction_converter = InstructionConverter(
            max_samples=config.max_samples.get("instruction", 0),
            llm_generator=self.llm_generator if config.enable_llm else None,
            sample_check_ratio=config.sample_check_ratio,
            review_output=config.review_output_path(),
            verify_code_exec=config.verify_code_exec,
            llm_max_attempts=config.llm_max_attempts,
        )
        self.cot_converter = COTConverter(
            max_samples=config.max_samples.get("cot", 0),
            llm_generator=self.cot_llm_generator if config.enable_llm else None,
            sample_check_ratio=config.sample_check_ratio,
            distribution_tolerance=config.cot_distribution_tolerance,
            review_output=config.review_output_path(cot=True),
            verify_code_exec=config.verify_code_exec,
            llm_max_attempts=config.llm_max_attempts,
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
                model=config.quality_model_resolved(),
                min_per_dimension=config.quality_min_per_dimension,
                min_total=config.quality_min_total,
                max_workers=config.max_workers,
                provider=config.llm_provider,
                base_url=config.llm_base_url(),
            )
            logger.info("质量评分已启用 (%s/%s, 阈值: 单项≥%d, 总分≥%d)",
                        config.llm_provider, config.quality_model_resolved(),
                        config.quality_min_per_dimension, config.quality_min_total)
        elif config.quality_enabled:
            logger.warning("未找到 %s API key，质量评分已跳过", config.llm_provider)

    def _init_llm_generator(self):
        """初始化LLM生成器(按 llm_provider 选端点: deepseek 默认)"""
        if not self.config.enable_llm:
            return None
        return self._build_generator(self.config.llm_model(),
                                     self.config.llm_max_tokens)

    def _build_generator(self, model: str, max_tokens: int):
        """按当前 provider 构造生成器(DeepSeek / DashScope 共用同一实现)"""
        provider = self.config.llm_provider
        if provider != "deepseek":
            raise ValueError("文本链只支持 deepseek(百炼已停用); "
                             "dashscope 仅用于 OCR 视觉链, 收到: %s" % provider)
        from .llm.dashscope_generator import DashScopeGenerator
        return DashScopeGenerator(
            api_key=self.config.api_key(),
            model=model,
            temperature=self.config.llm_temperature,
            max_tokens=max_tokens,
            provider=provider,
            base_url=self.config.llm_base_url(),
        )

    def _init_cot_llm_generator(self, base_generator):
        """COT 转换器专用生成器：llm_cot_model 指定时用该模型实例，否则与 instruction 共用 base_generator"""
        if not self.config.enable_llm or not self.config.llm_cot_model:
            return base_generator
        return self._build_generator(
            self.config.llm_cot_model,
            self.config.llm_cot_max_tokens or self.config.llm_max_tokens,
        )

    # ================= 主流程 =================
    def run(self):
        """运行完整pipeline"""
        from tqdm import tqdm

        logger.info("=" * 60)
        logger.info("[Pipeline] 文件处理管道启动")
        logger.info("=" * 60)

        # 1. 扫描文件（排序保证可复现）
        files = self._scan_files()
        self.stats.total_files = len(files)
        self.files_skipped_resume = 0

        # 1b. resume：整体跳过两类来源（读取/OCR/评分/转换都不再做）——
        #     ① 已产出记录的来源（各数据集来源并集）；
        #     ② 上轮在清洗/过滤阶段被"确定性"丢弃的来源（含 SimHash 近重复）：
        #        重跑结果不变，且单独重跑会因判重上下文丢失而放行、污染数据集。
        #     质检(quality 阶段)淘汰不在此列——API 限流等瞬时错误需要重跑救回。
        if self.config.resume and files:
            done: set = set()
            for name in self.config.output_types:
                done |= self.formatter.read_sources(name)
            done |= self._previous_deterministic_rejections()
            if done:
                kept = [f for f in files if os.path.basename(f) not in done]
                self.files_skipped_resume = len(files) - len(kept)
                files = kept
                logger.info("[Resume] 跳过 %d 个来源（已产出或确定性丢弃），待处理 %d 个",
                            self.files_skipped_resume, len(files))

        if not files:
            if self.files_skipped_resume:
                logger.info("[Resume] 全部 %d 个来源均已处理过，无需重跑",
                            self.files_skipped_resume)
            else:
                logger.warning("[WARN] 未找到任何可处理的文件")
            return

        logger.info("[Files] 待处理 %d 个文件", len(files))

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

            # 4.5 困惑度过滤（免费预筛，启用时）
            if self.ppl_filter is not None:
                cleaned_results = self._ppl_check(cleaned_results)

            # 5. 质量评分 + 过滤（启用时）
            if self.quality_scorer is not None:
                cleaned_results = self._quality_check(cleaned_results)

            # 6. 转换并输出
            converters = {
                "pretrain": self.pretrain_converter,
                "instruction": self.instruction_converter,
                "cot": self.cot_converter,
            }
            for name in ("pretrain", "instruction"):
                if name in self.config.output_types:
                    self._convert_and_output(name, cleaned_results, converters[name])
            # COT 以 SFT 产出为**唯一素材**(样本驱动) -> 必须在 instruction 阶段之后单独跑
            if "cot" in self.config.output_types:
                self._convert_cot(cleaned_results)

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
                    # HTML 结构化清洗后再走通用清洗（去界面残留/URL/断行等，
                    # 质检实测 wiki 下载件带导航行与 [编辑] 标记）
                    clean = self.text_cleaner.clean(self.html_cleaner.clean(result.text))
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

    # ================= 困惑度过滤 =================
    def _ppl_check(self, results: List[ReaderResult]) -> List[ReaderResult]:
        """困惑度过滤: 免费的本地预筛闸, 只保留 ppl <= 阈值的文档。
        扫码模式(threshold=None)全部通过, 只收集分数供标定。"""
        if not results:
            return []
        ppl_filter = self.ppl_filter
        assert ppl_filter is not None
        from tqdm import tqdm
        passed = []
        pbar = tqdm(total=len(results), desc="[PPL] 困惑度过滤", unit="text",
                    ncols=70, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
        for r in results:
            try:
                out = ppl_filter.check(r.text)
            except Exception as exc:  # noqa: BLE001
                # 单条打分异常(偶发 OOM/超长文本等)不判死: 语言自然度拿不准就放行,
                # 但必须告警 + 计数, 供人工复查(见 run_report.ppl_filter.score_errors)
                logger.warning("[PPL] 打分异常, 放行不淘汰: %s (%s)",
                               os.path.basename(r.filepath), exc)
                self.stats.ppl_errors += 1
                passed.append(r)
                pbar.update(1)
                continue
            pbar.update(1)
            if out.passed:
                passed.append(r)
            else:
                logger.info("  PPL 淘汰: %s (ppl %.1f > %.1f)",
                            os.path.basename(r.filepath), out.ppl, ppl_filter.threshold)
                self._record_rejection(os.path.basename(r.filepath), "ppl",
                                       f"ppl_{out.ppl:.1f}", r.text)
                self.stats.texts_filtered += 1
                self.stats.ppl_filtered += 1
        pbar.close()
        logger.info("[PPL] 困惑度过滤: %d -> %d (淘汰 %d, 打分异常放行 %d)",
                    len(results), len(passed), self.stats.ppl_filtered,
                    self.stats.ppl_errors)
        return passed

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

        exempt_sources = self._load_quality_exemptions()
        passed_results = []
        for i, (text, score, source) in enumerate(scored):
            if score.error:
                # 调用失败(网络/限流/超时)或响应无法解析: 不是"低质", 单独留证待重跑救回
                logger.warning("  评分异常(不作低质淘汰, 待重跑救回): %s - %s", source, score.error)
                self._record_api_error(source, score.error, text)
                self._record_rejection(source, "quality_api_error", score.error, text)
                self.stats.quality_api_errors += 1
                continue
            if score.passed or source in exempt_sources:
                if source in exempt_sources and not score.passed:
                    # 人工复核决定保留(如"结尾自然收束"判据误伤节选型来源): 放行但留证
                    logger.info("  人工豁免(经人工复核保留): %s (总分 %d/25) - %s",
                                source, score.total_score,
                                "; ".join(score.issues[:2]) or "")
                    self._record_quality_exemption(source, score, text)
                    self.stats.quality_exempted += 1
                passed_results.append(results[i])
            else:
                logger.info("  淘汰: %s (总分 %d/25) - %s",
                            source, score.total_score,
                            "; ".join(score.issues[:2]) or "")
                self._record_rejection(
                    source, "quality",
                    "low_score: " + ("; ".join(score.issues[:2]) or f"总分 {score.total_score}/25"),
                    text)

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
        """转换并输出数据集

        生成阶段（instruction / cot 的 LLM 通道）是网络等待型：一份语料 ~8.7 次调用、
        每次 25~120 秒，串行时服务端并发余量完全用不上（实测并发 8 路墙钟 6.6x，
        见 data/_pipeline 并发探针结论）。故按 config.llm_max_concurrency 做
        **来源级并发**：多份语料同时在飞，每份内部仍按块顺序生成 ——
        单份样本内容与串行完全一致，只是等待时间重叠。

        写盘集中在主线程（_write_lock 内）：formatter 的去重指纹集合非线程安全，
        写盘集中化也是 resume"来源级跳过"判据正确的前提。

        记录顺序可能因完成先后与串行不同（**样本集合一致**），换来的是已完成的
        来源随时落盘 —— 中途断网/被中断后 resume 不必重跑已产出的部分。
        """
        from tqdm import tqdm

        existing_sources = self.formatter.read_sources(name) if self.config.resume else set()
        if existing_sources:
            logger.info("[Output] [%s] resume: 已有 %d 个来源，跳过", name, len(existing_sources))

        pending = [r for r in results if os.path.basename(r.filepath) not in existing_sources]
        workers = self._generation_workers(converter, len(pending))

        total_samples = 0
        total_generated = 0
        written_sources = 0
        failures = 0
        empty_sources = 0

        pbar = tqdm(total=len(results), desc=f"[Output] {name}", unit="file",
                    ncols=70, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
        pbar.update(len(results) - len(pending))

        def consume(result, samples):
            nonlocal total_samples, total_generated, written_sources, empty_sources
            generated, written = self._write_converted(name, samples)
            total_generated += generated
            total_samples += written
            if written:
                written_sources += 1
            elif not generated:
                # 转换器内部对 LLM 调用失败是逐块 try/except(不冒泡) → 该份产 0 样本。
                # 断网/限流最常见的就是这种形态: 必须单独计量, 否则"静默少语料"。
                empty_sources += 1
                logger.warning("[Output] [%s] %s 未产出样本(LLM 调用失败或硬校验全剔), "
                               "未写入输出, resume 重跑会补",
                               name, os.path.basename(result.filepath))
            pbar.update(1)

        def on_failure(result, exc):
            nonlocal failures
            failures += 1
            logger.error("转换失败 [%s] %s: %s", name, os.path.basename(result.filepath), exc)
            pbar.update(1)

        if workers <= 1:
            for result in pending:
                try:
                    samples = converter.convert(result)
                except Exception as e:            # noqa: BLE001 - 单份失败不得中断整轮
                    on_failure(result, e)
                    continue
                consume(result, samples)
        else:
            logger.info("[Output] [%s] 来源级并发 %d 路（待生成 %d 份）",
                        name, workers, len(pending))
            with ThreadPoolExecutor(max_workers=workers,
                                    thread_name_prefix=f"gen-{name}") as ex:
                futures = {ex.submit(converter.convert, r): r for r in pending}
                for fut in as_completed(futures):
                    result = futures[fut]
                    try:
                        samples = fut.result()
                    except Exception as e:        # noqa: BLE001
                        on_failure(result, e)
                        continue
                    consume(result, samples)
        pbar.close()

        if failures or empty_sources:
            logger.warning("[Output] [%s] 生成异常合计 %d 份（硬失败 %d / 空产出 %d）"
                           "—— 均未写入输出，resume 重跑会补",
                           name, failures + empty_sources, failures, empty_sources)
        self.stats.generation_failures += failures
        self.stats.generation_empty += empty_sources

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
        # sources = 累计口径（resume 时本次可能只有个位数来源，若直接写本次值
        # 会把 metadata 里历史的 244+ 覆盖掉）；本次新增另记 sources_this_run
        self.formatter.write_metadata(
            name,
            sources=len(self.formatter.read_sources(name)),
            sources_this_run=written_sources,
            generated_at=datetime.now().isoformat(timespec="seconds"),
            elapsed_seconds=round(time.time() - self._t0, 1),
            config={
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "enable_llm": self.config.enable_llm,
                "quality_enabled": self.quality_scorer is not None,
                "quality_api_errors": self.stats.quality_api_errors,
            "quality_exempted": self.stats.quality_exempted,
            "quality_exempted_path": os.path.basename(self.quality_exempted_path),
                "quality_api_error_path": os.path.basename(self.quality_api_error_path),
                "generation_workers": workers,
                "llm_max_concurrency": self.config.llm_max_concurrency,
                "resume": self.config.resume,
            },
        )
        logger.info("[Output] [%s] %d records (%d sources) -> %s",
                    name, total_samples, written_sources, output_path)

    def _convert_cot(self, cleaned_results) -> None:
        """COT 阶段: 以 SFT 产出为**唯一素材**(样本驱动), 按来源分组生成

        - 素材 = instruction/data.jsonl 里的问答对本身(不得自造); 判类只在样本层:
          属于四类 -> 生成; 明确不属于四类 -> 不进 COT(计数留证); 判类难以区分 ->
          导出待人工审核清单, 不硬塞任何一类。
        - 语料原文: 本轮清洗结果优先, 缺失回退读取缓存(不重读文件、不重复 OCR),
          用于"回原文搜据"拼 context。
        - 并发与写盘纪律与生成阶段一致: 来源级并发, 写盘集中在主线程。
        - 生成结束后四类分布**只统计只报告**(不裁剪、不改判、不删样本)。
        """
        from tqdm import tqdm

        name = "cot"
        if getattr(self.cot_converter, "llm_generator", None) is None:
            logger.warning("[COT] 未启用 LLM 生成器(enable_llm=false) —— 思维链无法生成, 跳过 COT 数据集")
            self.formatter.write_metadata(name, status="skipped",
                                          reason="需要 LLM 生成器(enable_llm=true)")
            return
        sft_path = self.formatter.get_path("instruction")
        if not os.path.exists(sft_path):
            logger.error("[COT] 未找到 SFT 产出 %s —— COT 以 SFT 样本为素材, 请先跑 instruction 阶段",
                         sft_path)
            self.formatter.write_metadata(name, status="skipped",
                                          reason="未找到 SFT 产出(需先跑 instruction 阶段)")
            return

        records: List[Dict] = []
        with open(sft_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not records:
            logger.warning("[COT] SFT 产出无记录, 无可生成素材")
            self.formatter.write_metadata(name, status="skipped", reason="SFT 产出无记录")
            return

        by_source: Dict[str, List[Dict]] = {}
        for r in records:
            src = os.path.basename(str(r.get("source") or "").strip())
            if src:
                by_source.setdefault(src, []).append(r)
        logger.info("[COT] 素材: %d 条 SFT 样本 / %d 个来源 -> 逐来源判类并生成",
                    len(records), len(by_source))

        # 语料原文: 本轮清洗结果优先 -> 读取缓存兜底(按文件名对齐)
        text_by_name = {}
        for res in cleaned_results:
            if res.success and res.text:
                text_by_name[os.path.basename(res.filepath)] = res.text
        path_by_name = {}
        for fp in self._scan_files():
            path_by_name.setdefault(os.path.basename(fp), fp)

        existing = self.formatter.read_sources(name) if self.config.resume else set()
        pending = [s for s in by_source if s not in existing]
        if existing:
            logger.info("[Output] [%s] resume: 已有 %d 个来源，跳过", name, len(existing))

        workers = self._generation_workers(self.cot_converter, len(pending))
        total_samples = total_generated = written_sources = empty_sources = failures = 0

        pbar = tqdm(total=len(by_source), desc=f"[Output] {name}", unit="src",
                    ncols=70, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
        pbar.update(len(by_source) - len(pending))

        def corpus_for(src_name: str) -> str:
            if src_name in text_by_name:
                return text_by_name[src_name]
            fp = path_by_name.get(src_name)
            if fp is not None and getattr(self, "cache", None) is not None:
                cached = self.cache.get(fp)
                if cached is not None:
                    return cached.get("text", "") or ""
            logger.warning("[COT] 语料原文不可得(%s): 该来源的思维链只能引用样本自带材料", src_name)
            return ""

        def consume(src_name, samples):
            nonlocal total_samples, total_generated, written_sources, empty_sources
            generated, written = self._write_converted(name, samples)
            total_generated += generated
            total_samples += written
            if written:
                written_sources += 1
            elif not generated:
                empty_sources += 1
                logger.warning("[Output] [%s] %s 未产出样本(判类未命中/校验全剔/调用失败), "
                               "未写入输出, resume 重跑会补", name, src_name)
            pbar.update(1)

        def on_failure(src_name, exc):
            nonlocal failures
            failures += 1
            logger.error("COT 转换失败 [%s] %s: %s", name, src_name, exc)
            pbar.update(1)

        def run_one(src_name):
            return self.cot_converter.convert_sft_batch(
                src_name, by_source[src_name], corpus_for(src_name),
                corpus_type=os.path.splitext(src_name)[1].lstrip("."))

        if workers <= 1:
            for src_name in pending:
                try:
                    samples = run_one(src_name)
                except Exception as e:            # noqa: BLE001 - 单份失败不得中断整轮
                    on_failure(src_name, e)
                    continue
                consume(src_name, samples)
        else:
            logger.info("[COT] 来源级并发 %d 路（待生成 %d 份）", workers, len(pending))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gen-cot") as ex:
                futures = {ex.submit(run_one, s): s for s in pending}
                for fut in as_completed(futures):
                    src_name = futures[fut]
                    try:
                        samples = fut.result()
                    except Exception as e:        # noqa: BLE001
                        on_failure(src_name, e)
                        continue
                    consume(src_name, samples)
        pbar.close()

        if failures or empty_sources:
            logger.warning("[Output] [%s] 生成异常合计 %d 份（硬失败 %d / 空产出 %d）"
                           "—— 均未写入输出，resume 重跑会补",
                           name, failures + empty_sources, failures, empty_sources)
        self.stats.generation_failures += failures
        self.stats.generation_empty += empty_sources
        if total_generated != total_samples:
            logger.info("[Output] [%s] 跳过与数据集中已有记录完全相同的样本 %d 条",
                        name, total_generated - total_samples)
        self.stats.cot_samples = total_samples

        # 四类分布: 只统计只报告(读回落盘结果, 口径=最终数据集)
        all_samples: List[Dict] = []
        out_path = self.formatter.get_path(name)
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        all_samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        self.cot_converter.report_distribution(all_samples)

        self.formatter.write_metadata(
            name,
            sources=len(self.formatter.read_sources(name)),
            sources_this_run=written_sources,
            generated_at=datetime.now().isoformat(timespec="seconds"),
            elapsed_seconds=round(time.time() - self._t0, 1),
            note="素材 = instruction/data.jsonl 的 SFT 问答对(样本驱动); 分布只统计不裁剪",
            stats=dict(self.cot_converter.stats),
            config={
                "enable_llm": self.config.enable_llm,
                "llm_max_concurrency": self.config.llm_max_concurrency,
                "sample_check_ratio": self.config.sample_check_ratio,
                "llm_max_attempts": self.config.llm_max_attempts,
                "distribution_tolerance": self.config.cot_distribution_tolerance,
                "resume": self.config.resume,
            },
        )
        logger.info("[Output] [%s] %d records (%d sources) -> %s",
                    name, total_samples, written_sources, out_path)

    def _generation_workers(self, converter, n_pending: int) -> int:
        """生成阶段的并发路数

        只在"LLM 生成链"上并发：它的耗时 99% 是等 API 响应。纯本地转换（pretrain
        切块）并发没有收益，只会增加写盘竞争风险 → 保持单路。
        路数取 config.llm_max_concurrency（此前该配置项从未被使用）。
        """
        if n_pending <= 1 or getattr(converter, "llm_generator", None) is None:
            return 1
        try:
            configured = int(self.config.llm_max_concurrency)
        except (TypeError, ValueError):
            configured = 1
        return max(1, min(configured, n_pending))

    def _write_converted(self, name: str, samples) -> tuple:
        """把一份来源的样本写入数据集（锁内串行）→ (生成条数, 实际写入条数)"""
        if not samples:
            return 0, 0
        with self._write_lock:
            return len(samples), self.formatter.write_batch(name, samples)


    # ================= 报告 =================
    def _record_rejection(self, source: str, stage: str, reason: str, text: str = ""):
        # 生成阶段并发时由主线程调用；写盘集中在 _write_lock 语义下进行
        """记录被丢弃的文本，便于事后诊断阈值/OCR质量"""
        self.rejections[reason] += 1
        if not self.config.save_rejected:
            return
        if self._rejected_file is None:
            os.makedirs(os.path.dirname(os.path.abspath(self.rejected_path)) or ".", exist_ok=True)
            # resume 追加而非覆盖：历史"确定性丢弃"记录是下轮跳过判据的一部分
            mode = "a" if self.config.resume else "w"
            self._rejected_file = open(self.rejected_path, mode, encoding="utf-8")
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
        if self._api_error_file is not None:
            self._api_error_file.close()
            self._api_error_file = None

    def _previous_deterministic_rejections(self) -> set:
        """上轮 rejected.jsonl 中"确定性丢弃"（filter 阶段 / clean 空文本）的来源集合

        SimHash 近重复是典型：b.txt 上轮因与 a.txt 相似被滤，本轮若单独重跑，
        相似对象不在场 → 过滤放行 → 数据集混入近重复。quality 阶段淘汰
        不在此列（API 限流/瞬时错误可重跑救回，实测救回 4/5）。
        """
        out: set = set()
        if not os.path.exists(self.rejected_path):
            return out
        try:
            with open(self.rejected_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    src = rec.get("source")
                    if not src:
                        continue
                    stage = rec.get("stage")
                    if stage == "filter" or (stage == "clean"
                                             and rec.get("reason") == "empty_after_clean"):
                        out.add(os.path.basename(str(src)))
        except OSError as e:
            logger.warning("[Resume] 读取 %s 失败: %s", self.rejected_path, e)
        return out

    def _load_quality_exemptions(self) -> set:
        """人工豁免名单: 质检淘汰但已由人工复核决定保留的来源

        文件缺失/不可解析 → 按"无豁免"处理(绝不静默放行), 仅告警。
        """
        path = (self.config.quality_exempt_file or "").strip()
        if not path:
            return set()
        if not os.path.exists(path):
            alt = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", path))
            if os.path.exists(alt):
                path = alt
            else:
                return set()
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("[Quality] 人工豁免名单读取失败(%s): %s", path, e)
            return set()
        srcs = {os.path.basename(str(it.get("source", "")))
                for it in (data.get("sources") or []) if it.get("source")}
        if srcs:
            logger.info("[Quality] 人工豁免名单: %d 个来源 (%s)", len(srcs), path)
        return srcs

    def _record_quality_exemption(self, source: str, score, text: str = ""):
        """豁免留证: 记录被豁免来源的分数/维度/理由, 便于事后审计"谁被豁免、为什么" """
        try:
            if self._exempt_file is None:
                os.makedirs(os.path.dirname(os.path.abspath(self.quality_exempted_path)) or ".",
                            exist_ok=True)
                mode = "a" if (self.config.resume and os.path.exists(self.quality_exempted_path)) else "w"
                self._exempt_file = open(self.quality_exempted_path, mode, encoding="utf-8")
            self._exempt_file.write(json.dumps({
                "source": source,
                "total_score": getattr(score, "total_score", None),
                "dims": {k: getattr(score, k, None) for k in
                         ("content_completeness", "language_quality", "information_density",
                          "structure_clarity", "readability")},
                "issues": list(getattr(score, "issues", []) or []),
                "text_chars": len(text or ""),
                "decided_by": "人工复核(结尾自然收束判据豁免)",
            }, ensure_ascii=False) + "\n")
            self._exempt_file.flush()
        except Exception as e:
            logger.warning("豁免留证写入失败(%s): %s", source, e)

    def _record_api_error(self, source: str, error: str, text: str = ""):
        """记录"评分调用失败(网络/限流/超时)"的来源 —— 与真低质淘汰分开

        断网只在当轮少这份语料(不放行未评分的文本进数据集), 但它不进
        _previous_deterministic_rejections(那是 filter/clean 阶段的判据),
        故 resume 重跑会自动重评救回。
        """
        try:
            if self._api_error_file is None:
                os.makedirs(os.path.dirname(os.path.abspath(self.quality_api_error_path)) or ".", exist_ok=True)
                mode = "a" if self.config.resume else "w"
                self._api_error_file = open(self.quality_api_error_path, mode, encoding="utf-8")
            self._api_error_file.write(json.dumps({
                "source": source,
                "error": (error or "")[:300],
                "chars": len(text or ""),
                "ts": datetime.now().isoformat(timespec="seconds"),
            }, ensure_ascii=False) + "\n")
            self._api_error_file.flush()
        except OSError as e:
            logger.warning("记录评分异常来源失败: %s", e)

    def _write_run_report(self, files: List[str], cleaned_results: List[ReaderResult]):
        skipped = getattr(self, "files_skipped_resume", 0)
        report = {
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.time() - self._t0, 1),
            "total_files": len(files) + skipped,
            "files_skipped_resume": skipped,
            "files_processed": self.stats.files_processed,
            "files_failed": self.stats.files_failed,
            "cache": self.cache.summary(),
            "texts_cleaned": self.stats.texts_cleaned,
            "texts_filtered": self.stats.texts_filtered,
            "ppl_filter": {
                "enabled": self.ppl_filter is not None,
                "threshold": self.ppl_filter.threshold if self.ppl_filter else None,
                "filtered": self.stats.ppl_filtered,
                "score_errors": self.stats.ppl_errors,
            },
            "rejection_reasons": dict(self.rejections.most_common()),
            # 断网/限流导致的评分失败: 与"真低质淘汰"分开, 按清单定点补跑即可
            "quality_api_errors": self.stats.quality_api_errors,
            "quality_exempted": self.stats.quality_exempted,
            "quality_exempted_path": os.path.basename(self.quality_exempted_path),
            "generation_failures": self.stats.generation_failures,
            "generation_empty": self.stats.generation_empty,
            "quality_api_error_path": os.path.basename(self.quality_api_error_path),
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