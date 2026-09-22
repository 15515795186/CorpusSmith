"""
读取结果缓存
把"读取"阶段（PDF解析 / HTML解析 / OCR API 调用）的结果持久化到磁盘，
使重跑时无需再次解析文件、无需重复付费调用 OCR，同时保证 --resume 真正生效。

缓存键 = 文件绝对路径 + mtime_ns + size，文件一旦变化自动失效。
只缓存成功结果：失败（例如 API 抖动）不会被固化，下次运行会重试。
"""

import json
import os
import threading
from typing import Any, Dict, Optional


class ReadCache:
    """按文件指纹缓存 ReaderResult（JSONL，逐条落盘，崩溃不丢已完成的解析）"""

    def __init__(self, path: str, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        if self.enabled:
            self._load()

    # ---------- 内部工具 ----------
    @staticmethod
    def _stat_key(filepath: str) -> Optional[str]:
        try:
            st = os.stat(filepath)
        except OSError:
            return None
        return f"{os.path.abspath(filepath)}|{st.st_mtime_ns}|{st.st_size}"

    def _load(self):
        if not os.path.exists(self.path):
            return
        loaded = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 忽略损坏行（例如上次写入被中断）
                key = entry.get("key")
                if key:
                    self._entries[key] = entry
                    loaded += 1
        self._entries_loaded = loaded

    # ---------- 读写 ----------
    def get(self, filepath: str) -> Optional[Dict[str, Any]]:
        """命中则返回缓存记录（含 text / file_type / metadata），否则 None"""
        if not self.enabled:
            return None
        key = self._stat_key(filepath)
        if key is None:
            return None
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def put(self, filepath: str, text: str, file_type: str,
            metadata: Optional[dict] = None):
        """写入一条成功的读取结果（立即落盘）"""
        if not self.enabled:
            return
        key = self._stat_key(filepath)
        if key is None:
            return
        entry = {
            "key": key,
            "path": os.path.abspath(filepath),
            "file_type": file_type,
            "text": text,
            "metadata": metadata or {},
        }
        with self._lock:
            self._entries[key] = entry
            os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def summary(self) -> str:
        if not self.enabled:
            return "disabled"
        return f"命中 {self.hits} / 未命中 {self.misses} (缓存条目 {len(self._entries)})"

    def clear(self):
        with self._lock:
            self._entries.clear()
            if os.path.exists(self.path):
                os.remove(self.path)