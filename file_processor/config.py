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
    # 默认输入 = **项目最终语料**: 第三版爬取语料(data/爬取语料, 8 类 × 3 = 24 份)
    # 初版(400 份)与第二版(40 份)语料已删除(曾作测试用例); 换输入用 --input-dir / --files
    # 或 JSON 配置覆盖(单文件用 files 列表, 目录用 input_dirs)
    input_dirs: dict = field(default_factory=lambda: {
        "pdf_download": "data/爬取语料/文本pdf英文版",
        "pdf_download_cn": "data/爬取语料/文本pdf中文版",
        "pdf_photo": "data/爬取语料/版式pdf英文版",
        "pdf_photo_cn": "data/爬取语料/版式pdf中文版",
        "word": "data/爬取语料/word",
        "html_source": "data/爬取语料/源码html",
        "screenshot": "data/爬取语料/截图html",
        "txt": "data/爬取语料/txt",
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
    enable_llm: bool = True
    # 文本 LLM: 走 DeepSeek 官方 OpenAI 兼容端点(百炼欠费后文本链已全量切换)
    llm_provider: str = "deepseek"          # 文本链供应商标识(当前仅 deepseek)
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/"
    deepseek_model: str = "deepseek-flash"
    # DashScope: 仅保留给 OCR 视觉链(qwen-vl-plus), 不再参与文本生成
    dashscope_api_key: str = ""
    llm_cot_model: str = ""                # COT 转换器专用模型; 留空则与主模型共用
    llm_cot_max_tokens: int = 0            # COT 专用输出上限; 0 = 跟随 llm_max_tokens(deepseek-r1 思考链偏长, 建议 3000-4000)
    llm_max_attempts: int = 3              # 拒绝采样(失败重试制): LLM 生成后硬校验不过则重试的上限; 1=不重试
    llm_max_concurrency: int = 8
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096            # DeepSeek 为推理型模型, 思考 token 也计入上限, 预算过小会把可见输出挤空

    # ========== 样本级校验(转换后,从严) ==========
    # COT 四类分布: 只统计、只报告, 不裁剪也不改判(实事求是 —— 差额只能在素材层解决)。
    # tolerance 仅用于达标判定与告警(《数据集介绍.md》4.5 ±5%)。
    cot_distribution_tolerance: float = 0.05
    sample_check_ratio: float = 0.1           # 转换后 LLM 语义抽检比例(0=关闭; 抽检幻觉/跳步/类别匹配)
    verify_code_exec: bool = False            # 代码题运行验证(有安全门槛, 默认关)
    review_output: str = ""                 # 争议样本清单: 留空则跟随 output_dir(见 review_output_path)                   # 争议/未决样本导出路径(JSONL, 供人工双人审核); 空=仅记日志

    # ========== LLM 质量评分 ==========
    quality_enabled: bool = True
    quality_model: str = ""                # 留空 = 跟随当前 provider 的主模型
    quality_min_per_dimension: int = 3
    quality_min_total: int = 18
    # 人工豁免名单: 质检淘汰、但经人工复核判定应保留的来源(典型: "结尾自然收束"
    # 判据误伤节选型来源)。JSON 结构 {"sources": [{"source": "<文件名>", "reason": "...",
    # "scores": {...}}]}; 命中的来源即使质检未过也放行, 但逐条留证(quality_exempted.jsonl
    # + run_report.quality_exempted), 便于审计"谁被豁免、为什么"。
    quality_exempt_file: str = "data/_pipeline/人工豁免_结尾判据.json"

    # ========== 困惑度过滤(免费预筛闸, 可选) ==========
    # 位置: 启发式过滤之后、LLM 质量评分之前。PPL 高 = 语言不自然(机器翻译腔/OCR噪声), 淘汰 ppl > 阈值
    enable_ppl_filter: bool = True       # 已开闸: 阈值取绝对异常闸(非分位), 标定/决策过程见 data/_pipeline/PPL扫码留档.md
    ppl_model: str = "Qwen/Qwen2.5-0.5B" # 本地小 LM, 从 HUGGINGFACE_HUB_CACHE 缓存加载
    ppl_threshold: float = 25.0          # **极端异常闸**: 只拦机器翻译腔/乱码级异常(全量 80 份实测零淘汰);
                                         # 不用分位阈值 —— q95(16.94) 会连正常语料一起切(含新爬英文正文页/政务清单/规划), 见 data/_pipeline/PPL扫码留档.md; 0 = 扫码模式(只统计不淘汰)
    ppl_quantile: float = 0.95           # 仅扫码模式(标定/统计)使用; 生产阈值已改为上面的绝对异常闸, 不再按分位淘汰
    ppl_cache_dir: str = "D:/model"      # PPL 模型缓存根(与 data/_pipeline/scan_ppl_distribution.py 同源); 留空 = 跟随 HUGGINGFACE_HUB_CACHE/默认缓存

    # ========== 并发 ==========
    max_workers: int = 4

    # ========== 输出控制 ==========
    output_types: list = field(default_factory=lambda: ["pretrain", "instruction", "cot"])
    max_samples: dict = field(default_factory=lambda: {
        "pretrain": 0,
        "instruction": 0,
        "cot": 0,
    })
    resume: bool = True           # True: 不截断输出，跳过已写入的来源文件
    save_rejected: bool = True     # 保存被过滤/淘汰的文本，便于诊断阈值
    rejected_max_chars: int = 5000

    # ========== 读取缓存 ==========
    use_read_cache: bool = True    # 缓存"读取"结果（OCR/解析），重跑无需重复解析
    cache_dir: str = ".cache"

    # ========== 语言 ==========
    target_languages: list = field(default_factory=lambda: ["zh", "en"])
    keep_unknown_language: bool = True


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

    # ---------- 供应商解析: 文本 LLM 与 OCR(视觉) 各一套 key ----------
    def llm_model(self) -> str:
        """文本生成模型名(文本链已全量切 DeepSeek)"""
        return self.deepseek_model or "deepseek-flash"

    def llm_base_url(self) -> str:
        """文本链的 OpenAI 兼容端点"""
        return self.deepseek_base_url or "https://api.deepseek.com/"

    def api_key(self) -> str:
        """文本链 key（配置优先，其次环境变量 DEEPSEEK_API_KEY）"""
        return self.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")

    def ocr_api_key(self) -> str:
        """OCR(视觉链 qwen-vl)专用 key —— 与文本 LLM 的 provider 无关"""
        return self.dashscope_api_key or os.getenv("DASHSCOPE_API_KEY", "")

    def cot_model(self) -> str:
        """COT 专用模型(llm_cot_model 留空则跟随主模型)"""
        return self.llm_cot_model or self.llm_model()

    def review_output_path(self, cot: bool = False) -> str:
        """争议样本清单路径: 配置留空时跟随 output_dir(COT 单独一个文件)"""
        if self.review_output:
            return self.review_output
        name = "review_output_cot.jsonl" if cot else "review_output.jsonl"
        return os.path.join(self.output_dir, name)

    def quality_model_resolved(self) -> str:
        """质量评分模型: 配置留空时跟随文本链主模型"""
        return self.quality_model or self.llm_model()