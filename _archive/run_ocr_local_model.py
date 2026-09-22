import os
import torch
from transformers import AutoModel, AutoTokenizer

# 模型路径
model_name = "baidu/Unlimited-OCR"
cache_dir = r"D:\model"

# ── 加载 tokenizer ──
print("加载 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True,
)

# ── 加载模型（全放 GPU，需要先释放显存） ──
print("加载模型（bfloat16）...")
model = AutoModel.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
)

model = model.eval().cuda()

# 如果上面 .cuda() 报 OOM（显存不足），换成下面这行：
# model = model.eval()
# model = model.cuda()

print(f"✅ 模型加载成功！")
if torch.cuda.is_available():
    print(f"   显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ============================================================
# ⚡ 单张图片推理
# ============================================================
image_path = "1001.png"  # ← 换成你的图片路径

if os.path.exists(image_path):
    print(f"\n正在识别: {image_path}")
    model.infer(
        tokenizer,
        prompt="<image>document parsing.",
        image_file=image_path,
        output_path="./output",
        base_size=1024, image_size=640, crop_mode=True,
        max_length=32768,
        no_repeat_ngram_size=35, ngram_window=128,
        save_results=True,
    )
    print("✅ 识别完成，结果保存在 ./output 目录")
else:
    print(f"\n⚠️ 未找到测试图片: {image_path}")



