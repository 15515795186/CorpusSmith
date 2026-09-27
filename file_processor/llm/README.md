# llm/ — 生成器适配层

统一封装"让 LLM 生成文本"的调用。文本链默认走 **DeepSeek**
（`https://api.deepseek.com/`，模型 `deepseek-flash`），OpenAI 兼容端点。

## 文件说明

| 文件 | 职责 |
|------|------|
| `base.py` | `BaseLLMGenerator` 抽象接口：`generate()` / `async_generate()` |
| `dashscope_generator.py` | 通用 OpenAI 兼容端点生成器（类名保留历史）：按 `provider` 选端点与密钥，`deepseek` / `dashscope` 共用同一实现；语义别名 `OpenAICompatGenerator` |

端点优先级：显式 `base_url` > `<PROVIDER>_BASE_URL` 环境变量 > 内置表。
`deepseek-flash` 是**推理型模型**：隐藏思考 token 也计入 `max_tokens`，
预算过小会把可见输出挤空（质量评分因此固定留 3072）。

## 谁在用

- `pipeline._build_generator()` 按 `config.llm_provider` 构造，`_init_llm_generator()` /
  `_init_cot_llm_generator()` 分别注入 instruction / cot 转换器；
- **COT 已是样本驱动**：链全部由 LLM 生成（不再只在"模板样本不足"时兜底），并以既有 SFT 问答对为素材；
- `quality_scorer.py` 现在也走同一套 OpenAI 兼容客户端（`provider` + `base_url` 透传），
  不再直接用 dashscope SDK；
- 生成器只返回字符串，提示词构造与解析都在转换器内（`_build_*_prompt` / `_parse_*`）。

## 与 OCR 的关系

`readers/ocr_reader.py`（`qwen-vl-plus` 视觉 OCR）**不走本层**，它直接用
`dashscope` SDK 的多模态接口 —— 那是"识别"不是"生成"，且 DeepSeek 无多模态模型。
该链的 key 单独取 `config.ocr_api_key()`，与文本链的 provider 无关。

## 依赖

- 内部：仅 `base.py`
- 外部：`openai`（兼容客户端）；OCR 另需 `dashscope`（延迟 import）
- key 来源：`ProcessorConfig` 的 `deepseek_api_key` / `dashscope_api_key`，
  环境变量与项目根 `.env` 的回退链在 `main.build_config()` 完成
