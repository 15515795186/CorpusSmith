# _archive（历史方案 / 已废弃文件）

这里的文件**不再被主流程引用**，保留仅为可追溯性。主流程统一使用
DashScope Qwen-VL 做 OCR（见 `file_processor/readers/ocr_reader.py`）。

| 文件 | 原位置 | 废弃原因 |
|------|--------|----------|
| `screenshot_reader.py` | `file_processor/readers/` | 本地 `baidu/Unlimited-OCR` 方案，未接入 pipeline，已被 `QwenVLOcrReader` 取代 |
| `ocr_lock.py` | `file_processor/utils/` | 仅被 `screenshot_reader.py` 使用 |
| `run_ocr_local_model.py` | 项目根目录 | 本地模型单图冒烟脚本（原 `run_ocr.py`），对应方案已废弃 |
| `hf_download_model.py` | 项目根目录 | 本地模型下载脚本；已按 `huggingface_hub >= 1.0` 修正参数，若重启本地模型方案可直接使用 |
| `1001.png` | 项目根目录 | 本地 OCR 测试图（与 `data/_pipeline/ocr_test.png` 重复） |
| `output/` | 项目根目录 | 无关的终端记录（`result.md`）与本地 OCR 可视化输出 |

如需彻底删除：`Remove-Item -Recurse _archive`（删除前请确认不需要回溯）。