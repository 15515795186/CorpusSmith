"""
全局配置管理
支持 dataclass 配置 + JSON 配置覆盖
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ProcessorConfig:
    # ========== 路径 ==========
    # 默认指向仓库内真实的 data/ 目录结构；可通过 JSON 配置或 CLI 覆盖
    input_dirs: dict = field(default_factory=lambda: {
        "pdf_download": "data/文本pdf",
        "pdf_download_cn": "data/文本pdf中文版",
        "pdf_photo": "data/版式pdf",
        "pdf_photo_cn": "data/版式pdf中文版",
        "word": "data/word",
        "html_source": "data/源码html",
        "screenshot": "data/截图html",
        "txt": "data/txt",
    })
    files: Optional[list] = None
    output_dir: str = "datasets"
    output_prefix: str = ""

    # ========== OCR（Qwen-VL API）==========
    ocr_model: str = "qwen-vl-plus"   # DashScope 视觉模型
    ocr_min_text_chars: int = 50   # 少于这个字符数触发OCR
    ocr_dpi: int = 300             # 扫描PDF渲染DPI
    ocr_max_retries: int = 3
    ocr_work_dir: str = "ocr_temp"  # 扫描PDF页面渲染的临时目录

    # ========== 清洗参数 ==========
    min_text_length: int = 30
    # 默认基本等于"不按长度丢弃整篇文档"：长文档由分块(chunk_size)处理，
    # 旧默认 32768 会把 60KB+ 的正常论文整篇淘汰（实测确有发生）
    max_text_length: int = 1_000_000
    # 去重只清"近乎复制"的文档（相似度≥0.90 且长度比≥0.6，真实语料标定：
    # 不同论文/不同高校简介页全部保留，白皮书再版与模板重复被识别）
    dedup_threshold: float = 0.90
    garbage_ratio_threshold: float = 0.3  # 乱码字符比例上限
    max_repeat_char_ratio: float = 0.3    # 单字符占比上限（防退化输出）

    # ========== 分块 ==========
    chunk_size: int = 1024
    chunk_overlap: int = 128

    # ========== LLM 生成（可选）==========
    enable_llm: bool = False
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-4o-mini"
    dashscope_api_key: str = ""
    dashscope_model: str = "qwen-max"
    llm_batch_size: int = 5
    llm_max_concurrency: int = 3
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048

    # ========== LLM 质量评分 ==========
    quality_enabled: bool = True
    quality_model: str = "qwen-plus"
    quality_min_per_dimension: int = 3
    quality_min_total: int = 20

    # ========== 并发 ==========
    max_workers: int = 4

    # ========== 输出控制 ==========
    output_types: list = field(default_factory=lambda: ["pretrain", "instruction", "cot"])
    max_samples: dict = field(default_factory=lambda: {
        "pretrain": 0,
        "instruction": 0,
        "cot": 0,
    })
    resume: bool = False           # True: 不截断输出，跳过已写入的来源文件
    save_rejected: bool = True     # 保存被过滤/淘汰的文本，便于诊断阈值
    rejected_max_chars: int = 5000

    # ========== 读取缓存 ==========
    use_read_cache: bool = True    # 缓存"读取"结果（OCR/解析），重跑无需重复解析
    cache_dir: str = ".cache"

    # ========== 语言 ==========
    target_languages: list = field(default_factory=lambda: ["zh", "en"])
    keep_unknown_language: bool = True

    # ========== 数据增强 ==========
    enable_cot_augmentation: bool = False  # 是否启用基于模板的COT自动生成

    @classmethod
    def from_json(cls, path: str) -> "ProcessorConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 只保留 dataclass 定义的字段
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self, redact: bool = False) -> dict:
        """导出配置；redact=True 时隐藏所有 API key（用于日志/运行报告）"""
        data = asdict(self)
        if redact:
            for key in list(data):
                if "api_key" in key and data[key]:
                    data[key] = "***redacted***"
        return data

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def get_input_dir(self, file_type: str) -> str:
        return self.input_dirs.get(file_type, "")

    def get_all_input_dirs(self) -> list:
        return [v for v in self.input_dirs.values() if v]

    def resolved_input_dirs(self) -> dict:
        """仅返回真实存在的输入目录，避免无效扫描"""
        return {k: v for k, v in self.input_dirs.items()
                if v and os.path.isdir(v)}

    def api_key(self) -> str:
        """返回可用的 DashScope API key（配置优先，其次环境变量）"""
        return self.dashscope_api_key or os.getenv("DASHSCOPE_API_KEY", "")