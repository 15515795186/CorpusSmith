# llm/ — 生成器适配层（可选路径）

统一封装"让 LLM 生成文本"的调用，供 **instruction/cot 转换器兜底生成训练样本**。
默认不启用（`enable_llm=False`）：当前所有样本来自规则提取，确定性可复现。

## 文件说明

| 文件 | 职责 |
|------|------|
| `base.py` | `BaseLLMGenerator` 抽象接口：`generate()` / `async_generate()` |
| `dashscope_generator.py` | 通义千问文本生成（默认 `qwen3.5-plus`），OpenAI 兼容 `/chat/completions`，重试+退避 |

## 谁在用

- `pipeline._init_llm_generator()` 按 `llm_provider` 构造并注入两个转换器；
- 转换器只在**模板样本不足**（instruction<20 / cot<10）时调用它补充；
- 生成器只返回字符串，提示词构造与 JSON 解析都在转换器内
  （`_build_instruction_prompt` / `_build_cot_prompt` + `_parse_*`）。

## 与质检/OCR 的关系（易混淆，注意）

`quality_scorer.py`（qwen-plus 打分）和 `readers/ocr_reader.py`（qwen-vl OCR）**不走本层**，
它们直接调 `dashscope` SDK——那两处是"评估/识别"，不是"生成训练样本"，
需要的参数（result_format、多模态消息体）也不同。

## 依赖

- 内部：仅 `base.py`
- 外部：`dashscope`（延迟 import）；OpenAI 兼容端点走同一 SDK（`dashscope_generator` 内建 base_url）
- key 来源：`ProcessorConfig` 的 `dashscope_api_key`，
  环境变量与项目根 `.env` 的回退链在 `main.build_config()` 完成
