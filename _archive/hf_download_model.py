"""下载本地 OCR 模型（历史方案，见 _archive/README.md）

注意：huggingface_hub >= 1.0 已移除 `local_dir_use_symlinks` 与 `resume_download`
参数（断点续传现在默认开启），旧写法会直接抛 TypeError。
"""

import argparse
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="下载 HuggingFace 模型到本地目录")
    parser.add_argument("--model", default="baidu/Unlimited-OCR", help="仓库 id")
    parser.add_argument("--cache-dir", default=r"D:\model", help="缓存目录")
    args = parser.parse_args()

    path = snapshot_download(repo_id=args.model, cache_dir=args.cache_dir)
    print(f"✅ 模型下载完成: {path}")


if __name__ == "__main__":
    main()