"""
JSONL 格式写入器

输出布局（与 README 一致）::

    datasets/
    ├── pretrain/
    │   ├── data.jsonl
    │   └── metadata.json
    ├── instruction/
    │   ├── data.jsonl
    │   └── metadata.json
    ├── cot/
    │   ├── data.jsonl
    │   └── metadata.json
    └── quality_report.json

重要：每次运行默认 **重建**（prepare 时清空目标文件），避免历史上
"追加写" 导致多次运行结果叠加、出现大量重复样本。
"""

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)


class JSONLFormatter:
    """按数据集类型写入 data.jsonl / metadata.json"""

    def __init__(self, output_dir: str, prefix: str = "", overwrite: bool = True):
        self.output_dir = output_dir
        self.prefix = prefix
        self.overwrite = overwrite
        self._prepared: Set[str] = set()
        os.makedirs(output_dir, exist_ok=True)

    # ---------- 路径 ----------
    def dataset_dir(self, name: str) -> str:
        return os.path.join(self.output_dir, name)

    def _filename(self, name: str) -> str:
        return os.path.join(self.dataset_dir(name), f"{self.prefix}data.jsonl")

    def get_path(self, name: str) -> str:
        return self._filename(name)

    def get_metadata_path(self, name: str) -> str:
        return os.path.join(self.dataset_dir(name), "metadata.json")

    def get_report_path(self) -> str:
        return os.path.join(self.output_dir, "quality_report.json")

    # ---------- 生命周期 ----------
    def prepare(self, name: str):
        """创建目录；overwrite=True 时清空已有输出（保证结果可复现）"""
        os.makedirs(self.dataset_dir(name), exist_ok=True)
        path = self._filename(name)
        if self.overwrite and os.path.exists(path):
            os.remove(path)
            logger.info("[Output] 已清空旧文件: %s", path)
        self._prepared.add(name)

    def prepare_all(self, names: Iterable[str]):
        for name in names:
            self.prepare(name)

    # ---------- 写入 ----------
    def write_batch(self, name: str, records: List[Dict[str, Any]]):
        if not records:
            return
        if name not in self._prepared:
            self.prepare(name)
        os.makedirs(self.dataset_dir(name), exist_ok=True)
        with open(self._filename(name), "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write(self, name: str, record: Dict[str, Any]):
        self.write_batch(name, [record])

    # ---------- 读取 / 查询 ----------
    def count_lines(self, name: str) -> int:
        path = self._filename(name)
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def exists(self, name: str) -> bool:
        return os.path.exists(self._filename(name))

    def read_sources(self, name: str) -> Set[str]:
        """读取已写入数据的来源文件名集合（供 --resume 跳过重复来源）"""
        path = self._filename(name)
        sources: Set[str] = set()
        if not os.path.exists(path):
            return sources
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                meta = rec.get("meta") or {}
                src = meta.get("source") or rec.get("source")
                if src:
                    sources.add(os.path.basename(str(src)))
        return sources

    # ---------- 元数据 ----------
    def write_metadata(self, name: str, **extra):
        os.makedirs(self.dataset_dir(name), exist_ok=True)
        metadata = {
            "dataset_name": name,
            "status": "completed",
            "records": self.count_lines(name),
            "data_file": os.path.basename(self._filename(name)),
        }
        metadata.update(extra)
        with open(self.get_metadata_path(name), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return metadata