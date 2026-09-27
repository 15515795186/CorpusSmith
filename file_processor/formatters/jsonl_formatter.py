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

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Set

from ..utils.lang_detect import detect_language

logger = logging.getLogger(__name__)


class JSONLFormatter:
    """按数据集类型写入 data.jsonl / metadata.json

    按《数据集介绍.md》3.2 的推荐字段, 对 SFT(instruction)输出补 ``id`` 与 ``language``:
    id 形如 ``sft-000001``(8.2 命名规范), language 取 lang_detect 的 zh/en/mixed/unknown。
    两个字段在写入时注入, **不参与去重指纹** —— 否则内容完全相同的样本会因 id
    不同被当成两条写进数据集。其余数据集(pretrain/cot)不注入, 保持原样。
    """

    # 需要注入通用字段的数据集 -> 字段名(其余数据集原样写出)
    _INJECT = {"instruction": ("id", "language"), "cot": ("id",)}
    # id 前缀(8.2: {数据集}-{序号})
    _ID_PREFIX = {"instruction": "sft", "pretrain": "pretrain", "cot": "cot"}

    def __init__(self, output_dir: str, prefix: str = "", overwrite: bool = True):
        self.output_dir = output_dir
        self.prefix = prefix
        self.overwrite = overwrite
        self._prepared: Set[str] = set()
        # 跨批次/跨运行的样本级精确去重（同一数据集内完全相同的记录只写一次）
        self._seen: Dict[str, Set[str]] = {}
        # 各数据集已分配的 id 序号(续跑时接续历史最大值)
        self._seq: Dict[str, int] = {}
        self.duplicates_skipped = 0
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
        self._seen[name] = set()
        self._seq[name] = 0
        if not self.overwrite and os.path.exists(path):
            # 续跑模式：载入已有记录的规范化指纹，保证新记录不与历史样本重复
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._seen[name].add(self._record_key(rec))
                    # 续跑: id 序号接续历史最大值, 避免与已有 id 撞号
                    rid = str(rec.get("id") or "")
                    tail = rid.rsplit("-", 1)[-1]
                    if tail.isdigit():
                        cls = str(rec.get("task_type") or "").strip() if name == "cot" else ""
                        key = "%s/%s" % (name, cls) if cls else name
                        self._seq[key] = max(self._seq.get(key, 0), int(tail))
            if not self._seq[name]:
                self._seq[name] = len(self._seen[name])
        self._prepared.add(name)

    def prepare_all(self, names: Iterable[str]):
        for name in names:
            self.prepare(name)

    # ---------- 写入 ----------
    @classmethod
    def _record_key(record_cls, record: Dict[str, Any]) -> str:
        """内容指纹: 忽略 _INJECT 注入的字段, 保证去重判据只取决于样本内容"""
        skip = {f for fields in record_cls._INJECT.values() for f in fields}
        raw = json.dumps({k: v for k, v in record.items() if k not in skip},
                         ensure_ascii=False, sort_keys=True)
        return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()

    @staticmethod
    def _sample_language(text: str) -> str:
        """样本语言(规范 3.2 的 language 取 zh/en 口径)

        中文样本常夹技术专有名词(ClickHouse/Loki/Elasticsearch、PDF/DOCX…), 字母数
        可能反超汉字数, 单纯按字符数会把中文样本判成 en(实测 runF 前 40 条就有一批)。
        故: 汉字数 >= 10 且占中英字符 >= 10% 即视为中文样本; 纯/主英文样本判 en;
        极短或符号为主的文本回退 lang_detect(可返回 zh/en/mixed/unknown)。
        """
        zh = len(re.findall(r"[\u4e00-\u9fff]", text))
        en = len(re.findall(r"[A-Za-z]", text))
        if zh >= 10 and zh >= 0.10 * (zh + en):
            return "zh"
        if en >= 10:
            return "en"
        return detect_language(text)

    def _enrich(self, name: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """按数据集补通用字段; 未登记的数据集原样返回

        - id: 写入序分配(单一写线程 -> 稳定), 已有 id 的记录不覆盖
        - language: 由 instruction + output 判定, 已有值不覆盖
        """
        fields = self._INJECT.get(name)
        if not fields:
            return record
        out = dict(record)
        if "id" in fields and not out.get("id"):
            # COT 的 id 带任务类型且**按类分别编号**(8.2: cot-{任务类型}-{序号})
            cls = str(out.get("task_type") or "").strip() if name == "cot" else ""
            key = "%s/%s" % (name, cls) if cls else name
            self._seq[key] = self._seq.get(key, 0) + 1
            if cls:
                out["id"] = "cot-%s-%06d" % (cls, self._seq[key])
            else:
                out["id"] = "%s-%06d" % (self._ID_PREFIX.get(name, name), self._seq[key])
        if "language" in fields and not out.get("language"):
            text = "%s\n%s" % (out.get("instruction") or "", out.get("output") or "")
            out["language"] = self._sample_language(text)
        return out

    def write_batch(self, name: str, records: List[Dict[str, Any]]) -> int:
        """写入一批记录，跳过数据集中已存在的完全相同记录；返回实际写入条数"""
        if not records:
            return 0
        if name not in self._prepared:
            self.prepare(name)
        os.makedirs(self.dataset_dir(name), exist_ok=True)
        seen = self._seen.setdefault(name, set())
        written = 0
        with open(self._filename(name), "a", encoding="utf-8") as f:
            for record in records:
                key = self._record_key(record)
                if key in seen:
                    self.duplicates_skipped += 1
                    continue
                seen.add(key)
                f.write(json.dumps(self._enrich(name, record), ensure_ascii=False) + "\n")
                written += 1
        return written

    def write(self, name: str, record: Dict[str, Any]) -> int:
        return self.write_batch(name, [record])

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