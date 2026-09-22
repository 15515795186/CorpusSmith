import os
import unittest

from file_processor.utils.cache import ReadCache
from tests._tempdir import cleanup, workspace_tempdir


class ReadCacheTest(unittest.TestCase):

    def setUp(self):
        self._tmp = workspace_tempdir()
        self.cache_path = os.path.join(self._tmp, "read_cache.jsonl")
        self.file = os.path.join(self._tmp, "sample.txt")
        with open(self.file, "w", encoding="utf-8") as f:
            f.write("hello world")

    def tearDown(self):
        cleanup(self._tmp)

    def test_put_then_get_hits_new_instance(self):
        cache = ReadCache(self.cache_path)
        cache.put(self.file, "hello world", "txt", {"encoding": "utf-8"})
        self.assertEqual(cache.hits, 0)

        reloaded = ReadCache(self.cache_path)
        entry = reloaded.get(self.file)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["text"], "hello world")
        self.assertEqual(entry["file_type"], "txt")
        self.assertEqual(reloaded.hits, 1)

    def test_changed_file_invalidates_cache(self):
        cache = ReadCache(self.cache_path)
        cache.put(self.file, "hello world", "txt")
        with open(self.file, "w", encoding="utf-8") as f:
            f.write("completely different and longer content")
        self.assertIsNone(cache.get(self.file))
        self.assertEqual(cache.misses, 1)

    def test_disabled_cache_is_noop(self):
        cache = ReadCache(self.cache_path, enabled=False)
        cache.put(self.file, "hello world", "txt")
        self.assertIsNone(cache.get(self.file))
        self.assertFalse(os.path.exists(self.cache_path))
        self.assertEqual(cache.summary(), "disabled")

    def test_corrupt_lines_are_ignored(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            f.write("{not json}\n")
        cache = ReadCache(self.cache_path)
        cache.put(self.file, "text", "txt")
        reloaded = ReadCache(self.cache_path)
        self.assertIsNotNone(reloaded.get(self.file))

    def test_clear_removes_file(self):
        cache = ReadCache(self.cache_path)
        cache.put(self.file, "text", "txt")
        cache.clear()
        self.assertFalse(os.path.exists(self.cache_path))
        self.assertIsNone(cache.get(self.file))


if __name__ == "__main__":
    unittest.main()