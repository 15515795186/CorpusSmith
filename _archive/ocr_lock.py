"""
OCR 模型加载全局锁
防止多线程同时加载模型导致 GPU OOM
"""
import threading

ocr_load_lock = threading.Lock()