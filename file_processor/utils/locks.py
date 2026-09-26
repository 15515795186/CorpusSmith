"""进程级写锁：并发（来源级）生成时，串行化"多线程写同一文件"的路径

背景：生成阶段(instruction/cot)是网络等待型 —— 一份语料 ~8.7 次 LLM 调用、
每次 25~120 秒，串行时服务端并发余量完全用不上(实测并发 8 路墙钟 6.6x)。
改为来源级并发后：

- 数据集输出文件：由 Pipeline 自己的 _write_lock 串行化。formatter 的去重
  指纹集合(_seen)不是线程安全的，且写盘集中化才能保证 resume 的"来源级跳过"
  判据正确 —— 见 pipeline._convert_and_output / _write_converted。
- 争议样本清单(review_output.jsonl)：转换器在 worker 线程内追加写，多线程
  同时 open(...,'a') 会让行交错 → 用 REVIEW_LOCK 串行化。

只锁"写"，不锁"算"：LLM 调用与文本处理全程在锁外并发。
"""
import threading

REVIEW_LOCK = threading.Lock()
