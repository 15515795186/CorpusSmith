"""
PPL(困惑度)过滤器 —— 流水线里的一道免费预筛闸门
================================================

位置: 清洗+启发式过滤(quality_filter)之后、LLM 质量评分(qwen-plus)之前。
目标: 在花 API 钱之前, 用本地小 LM 把"语言不自然"的文本拦掉
      (机器翻译腔 / OCR 噪声 / 乱码拼接), 语义层面的低质仍由后面的 LLM 评分把关。

语义: PPL 高 = 文本难预测 = 远离自然语言。
      保留 ppl <= 阈值; 淘汰 ppl > 阈值。

阈值两种模式:
  - 显式阈值(threshold > 0): 高于即淘汰
  - 扫码模式(threshold=None, 默认): 只统计不淘汰 —— 跑一批文本收集 PPL 分布,
    按分位数(ppl_quantile, 默认 0.90)标定阈值(calibrate_threshold), 再把阈值写进配置。
    绝不拍脑袋定绝对值: 政务公文与维基百科的 PPL 基准完全不同。

实现注记:
  - 延迟加载 transformers, 未启用时零 import 成本(与 dashscope generator 同模式)
  - 模型缓存根可显式传入(cache_dir / config.ppl_cache_dir), 不依赖外部 HUGGINGFACE_HUB_CACHE;
    加载失败 fail fast 并给出可执行的提示, 不静默回退成"全部放行"
  - 因果 LM 的 per-token 平均交叉熵 = loss, PPL = exp(loss), 由 labels=input_ids 直接得到
  - 单文本截断 2048 token(长文只评估前缀, 对"语言自然度"足够)
  - 不做 batch padding(填充 0 pad token 会污染 PPL)
"""

import logging
import math

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
_MAX_TOKENS = 2048
# 短于该长度直接放行: PPL 在极小样本上不稳定, 且 quality_filter 已拦住过短文本
_MIN_CHARS = 20


class PPLFilter:
    def __init__(self, model_name: str = DEFAULT_MODEL,
                 threshold: float | None = None,
                 quantile: float = 0.90,
                 cache_dir: str | None = None):
        self.model_name = model_name
        self.threshold = threshold          # None = 扫码模式(只统计)
        self.quantile = quantile            # 扫码模式自动定标分位数
        self.cache_dir = cache_dir          # 模型缓存根; None = 跟随 HUGGINGFACE_HUB_CACHE/默认缓存
        self._tokenizer = None
        self._model = None
        self.scores: list[float] = []       # 本批(或历年扫码)收集的 PPL 分数

    # ---------------- 模型加载(延迟) ----------------
    def _ensure_model(self):
        if self._model is not None:
            return
        # 模型文件用镜像(HF_ENDPOINT)预下载到本地缓存后, 离线加载:
        # from_pretrained 仍会联网 HEAD 检查远端更新, 主站超时时直接挂;
        # HF_HUB_OFFLINE=1 跳过一切网络检查, 纯用本地缓存
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        if self.cache_dir:
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", self.cache_dir)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kw = {"cache_dir": self.cache_dir} if self.cache_dir else {}
        log.info("[PPL] 加载模型 %s (cache_dir=%s) ...",
                 self.model_name, self.cache_dir or "默认缓存")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **kw)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kw)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"[PPL] 本地模型加载失败: {self.model_name} "
                f"(cache_dir={self.cache_dir or '默认缓存'}) —— 请确认 ppl_cache_dir / "
                "HUGGINGFACE_HUB_CACHE 指向已下载模型的缓存根; "
                f"原始错误: {exc}") from exc
        log.info("[PPL] 模型就绪")

    def close(self):
        """释放 GPU/显存(在 CUDA 可用时); CPU 上无操作"""
        try:
            if self._model is not None:
                self._model.cpu()
                self._model = None
        except Exception:  # noqa: BLE001
            pass

    # ---------------- PPL 计算 ----------------
    def _ppl_of(self, text: str) -> float:
        import torch
        self._ensure_model()
        assert self._tokenizer is not None and self._model is not None
        enc = self._tokenizer(
            text[: 8192], return_tensors="pt",
            truncation=True, max_length=_MAX_TOKENS)
        input_ids = enc.input_ids
        if input_ids.numel() == 0:
            return 0.0
        with torch.no_grad():
            out = self._model(input_ids=input_ids, labels=input_ids)
        loss = out.loss.item()
        if loss < 0 or not math.isfinite(loss):
            # exp(负) 理论不可能; 防御 padding/退化输出
            return float("inf")
        return math.exp(loss)

    # ---------------- 主入口 ----------------
    def check(self, text: str) -> "PPLResult":
        """单文本过滤。扫码模式(threshold=None)恒通过, 只收集分数。"""
        text = (text or "").strip()
        if len(text) < _MIN_CHARS:
            return PPLResult(True, 0.0, None)   # 过短放行
        ppl = self._ppl_of(text)
        self.scores.append(ppl)
        if self.threshold is not None and ppl > self.threshold:
            return PPLResult(False, ppl, f"ppl {ppl:.1f} > {self.threshold:.1f}")
        return PPLResult(True, ppl, None)

    # ---------------- 阈值标定 ----------------
    def calibrate_threshold(self, quantile: float | None = None) -> float:
        """按已收集分数的分位数定阈值(扫码模式收尾用)。
        语义: 淘汰最高 (1-q) 比例 —— 取第 ceil(n*q) 个样本(1起算)为界,
        n=10, q=0.9 时 idx=8 -> 第9个样本为界, 高于者淘汰(恰淘汰最高 10%)。"""
        import math
        q = quantile or self.quantile
        scores = sorted(self.scores)
        if not scores:
            raise ValueError("no scores collected, cannot calibrate")
        idx = min(max(int(math.ceil(len(scores) * q)) - 1, 0), len(scores) - 1)
        self.threshold = scores[idx]
        log.info("[PPL] 阈值标定: q=%s -> %.2f (n=%d)", q, self.threshold, len(scores))
        return self.threshold


class PPLResult:
    __slots__ = ("passed", "ppl", "reason")

    def __init__(self, passed: bool, ppl: float, reason):
        self.passed = passed
        self.ppl = ppl
        self.reason = reason

    def __repr__(self):
        return f"PPLResult(passed={self.passed}, ppl={self.ppl:.2f}, reason={self.reason})"