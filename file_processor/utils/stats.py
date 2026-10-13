"""
数据统计工具
"""

from collections import Counter
from typing import List


class StatsCollector:
    """收集处理统计数据"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_files = 0
        self.files_by_type = Counter()
        self.files_processed = 0
        self.files_failed = 0
        self.texts_cleaned = 0
        self.texts_filtered = 0
        self.ppl_filtered = 0          # 困惑度过滤淘汰的文本数
        self.ppl_errors = 0            # 困惑度打分异常(放行不淘汰)的文本数
        self.quality_api_errors = 0    # 质量评分调用失败(网络/限流)的来源数
        self.quality_exempted = 0       # 质检未过但经人工复核豁免保留的来源数
        self.generation_failures = 0    # 生成阶段转换失败的来源数(未写入输出, resume 会补)
        self.generation_empty = 0       # 生成阶段产 0 样本(多为 LLM 调用失败)的来源数
        self.duplicates_removed = 0
        self.cache_hits = 0
        self.pretrain_samples = 0
        self.instruction_samples = 0
        self.cot_samples = 0
        self.language_dist = Counter()
        self.total_tokens_est = 0
        self.errors: List[str] = []

    def add_error(self, filepath: str, error: str):
        self.errors.append(f"[{filepath}] {error}")

    def print_summary(self):
        print("\n" + "=" * 60)
        print("[Stats] 处理统计报告")
        print("=" * 60)
        print(f"  发现文件总数:      {self.total_files}")
        print("  按类型:")
        for ft, cnt in self.files_by_type.most_common():
            print(f"    - {ft}: {cnt}")
        print(f"  成功处理:          {self.files_processed}")
        print(f"  处理失败:          {self.files_failed}")
        print(f"  命中读取缓存:      {self.cache_hits}")
        print(f"  清洗后文本段数:    {self.texts_cleaned}")
        print(f"  质量过滤丢弃:      {self.texts_filtered}")
        print(f"  PPL 淘汰:          {self.ppl_filtered}")
        print(f"  PPL 打分异常放行:  {self.ppl_errors}")
        print(f"  去重移除:          {self.duplicates_removed}")
        print(f"  预计总 Token 数:    {self.total_tokens_est:,}")
        print(f"  语言分布:          {dict(self.language_dist.most_common())}")
        print("  ----")
        print(f"  预训练样本数:      {self.pretrain_samples}")
        print(f"  指令微调样本数:    {self.instruction_samples}")
        print(f"  COT 样本数:        {self.cot_samples}")
        if self.errors:
            print(f"  [WARN] 错误 ({len(self.errors)}):")
            for e in self.errors[-5:]:
                print(f"    - {e}")
        print("=" * 60)