"""
指令微调数据集转换器
- 模板模式: 从文档结构自动提取 QA 对
- LLM模式: 调用LLM生成多样化的指令-输出对
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional

from ..readers.base import ReaderResult
from ..utils.lang_detect import detect_language, contains_chinese, contains_english

logger = logging.getLogger(__name__)


class InstructionConverter:
    """转换为指令微调数据集格式"""

    def __init__(self, max_samples: int = 0, llm_generator=None):
        self.max_samples = max_samples
        self.llm_generator = llm_generator

    def convert(self, result: ReaderResult) -> List[Dict]:
        """将读取结果转换为指令样本列表"""
        if not result.success or not result.text:
            return []

        text = result.text.strip()
        samples = []

        # 先尝试用模板规则生成
        template_samples = self._generate_from_templates(text, result)
        samples.extend(template_samples)

        # 如果启用了LLM，用LLM补充更多样化的指令
        if self.llm_generator and len(samples) < 20:
            llm_samples = self._generate_with_llm(text, result)
            samples.extend(llm_samples)

        # 限制数量
        if self.max_samples > 0:
            samples = samples[:self.max_samples]

        return samples

    def _generate_from_templates(self, text: str, result: ReaderResult) -> List[Dict]:
        """从文本结构中用模板生成指令对"""
        samples = []
        source = os.path.basename(result.filepath)

        # 1. 从标题 + 后续内容生成"什么是XX"问题
        heading_pattern = re.compile(r'^#{1,3}\s+(.+)$', re.MULTILINE)
        headings = list(heading_pattern.finditer(text))

        for i, m in enumerate(headings):
            title = m.group(1).strip()
            # 找标题后的内容（直到下一个标题或结束）
            start = m.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            content = text[start:end].strip()

            if content and len(content) > 20:
                # 生成不同类型的指令
                for template in self._get_title_templates(title):
                    instruction, input_text, output = template
                    if output:
                        samples.append({
                            "instruction": instruction,
                            "input": input_text,
                            "output": output[:1000],  # 截断过长输出
                            "source": source,
                        })
                        if len(samples) >= 30:
                            return samples

        # 2. 从列表中提取
        list_pattern = re.compile(r'^[\-\*]\s+(.+)$', re.MULTILINE)
        list_items = list_pattern.findall(text)
        list_items = [item.strip() for item in list_items if item.strip()]

        if len(list_items) >= 3:
            # 查找列表前的上下文（找前面最近的标题或段落）
            context = self._find_list_context(text, list_items)
            if context:
                for template in self._get_list_templates(context, list_items):
                    instruction, input_text, output = template
                    if output:
                        samples.append({
                            "instruction": instruction,
                            "input": input_text,
                            "output": output[:2000],
                            "source": source,
                        })
                        if len(samples) >= 30:
                            return samples

        # 3. 从定义式句子中提取
        def_pattern = re.compile(r'^(.{2,30})(?:是指|指的是|即|是(?:一[种类])?|属于|意为)(.{10,200})', re.MULTILINE)
        for m in def_pattern.finditer(text):
            term = m.group(1).strip()
            definition = m.group(2).strip()
            if term and definition:
                samples.append({
                    "instruction": f"什么是{term}？",
                    "input": "",
                    "output": f"{term}是指{definition}",
                    "source": source,
                })
                if len(samples) >= 30:
                    return samples

        return samples

    def _get_title_templates(self, title: str) -> List[tuple]:
        """根据标题生成指令模板"""
        templates = []

        # 模板1: "什么是{title}"
        templates.append((
            f"什么是{title}？请详细介绍。", "", None
        ))

        # 模板2: "解释{title}的概念"
        templates.append((
            f"解释一下{title}的主要内容和核心概念。", "", None
        ))

        # 模板3: "总结{title}"
        templates.append((
            f"请总结关于「{title}」的核心要点。", "", None
        ))

        # 模板4: 英文兼容
        if contains_english(title):
            templates.append((
                f"What is {title}? Please explain in detail.", "", None
            ))
            templates.append((
                f"Summarize the key points about {title}.", "", None
            ))

        return templates

    def _get_list_templates(self, context: str, items: List[str]) -> List[tuple]:
        """根据列表项生成指令模板"""
        templates = []
        list_text = "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))

        # 模板1: 列举
        templates.append((
            f"请列举{context}。", "",
            f"以下是{context}的详细说明：\n{list_text}"
        ))

        # 模板2: 具体解释每条
        if len(items) >= 2:
            for i, item in enumerate(items):
                templates.append((
                    f"解释「{item}」的含义。", "",
                    item
                ))

        return templates

    def _find_list_context(self, text: str, items: List[str]) -> str:
        """查找列表项的上下文（前面最近的标题或句子）"""
        if not items:
            return "相关内容"

        first_item = items[0]
        idx = text.find(first_item)
        if idx < 0:
            return "相关内容"

        # 找前面的标题
        before = text[:idx]
        heading_m = re.findall(r'^#{1,3}\s+(.+)$', before, re.MULTILINE)
        if heading_m:
            return heading_m[-1].strip()

        # 找前面第一个句号/冒号前的文本
        sentences = re.split(r'[。：:]', before)
        if sentences:
            ctx = sentences[-1].strip()
            if len(ctx) > 2:
                return ctx

        return "相关内容"

    def _generate_with_llm(self, text: str, result: ReaderResult) -> List[Dict]:
        """调用LLM生成指令-输出对"""
        if not self.llm_generator:
            return []

        samples = []
        source = os.path.basename(result.filepath)
        language = detect_language(text)

        # 按长度切块，每块生成指令
        chunks = self._split_into_chunks(text, 1500)
        for chunk in chunks[:5]:  # 最多处理5块
            try:
                prompt = self._build_instruction_prompt(chunk, language)
                response = self.llm_generator.generate(
                    prompt,
                    system_prompt="你是一个专业的数据标注助手。请根据文本内容生成高质量的指令-响应对。"
                )

                parsed = self._parse_llm_output(response)
                for item in parsed:
                    item["source"] = source
                    samples.append(item)

            except Exception as e:
                logger.warning("LLM generation failed: %s", e)
                continue

        return samples

    def _split_into_chunks(self, text: str, max_chars: int) -> List[str]:
        """智能分块"""
        chunks = []
        paragraphs = text.split('\n\n')
        current = ""
        for para in paragraphs:
            if len(current) + len(para) > max_chars and current:
                chunks.append(current.strip())
                current = para
            else:
                current += "\n\n" + para if current else para
        if current:
            chunks.append(current.strip())
        return chunks

    def _build_instruction_prompt(self, text: str, language: str) -> str:
        """构建LLM生成指令的prompt"""
        if language == "zh":
            return f"""从以下文本中生成3-5个高质量的指令-响应对，覆盖不同的信息维度（总结、提取、问答、推理等）。

要求：
1. 指令要多样，覆盖不同信息维度
2. 答案必须严格基于文本内容，不编造
3. 输出JSON数组格式

文本内容：
{text}

请按以下格式输出（严格JSON数组）：
[
  {{"instruction": "xxx", "input": "", "output": "xxx"}},
  ...
]"""
        else:
            return f"""Generate 3-5 high-quality instruction-response pairs from the following text.
Cover multiple dimensions: summarization, extraction, Q&A, reasoning, etc.

Requirements:
1. Diverse instructions
2. Answers must be strictly based on the text
3. Output in JSON array format

Text:
{text}

Output format (strict JSON array):
[
  {{"instruction": "xxx", "input": "", "output": "xxx"}},
  ...
]"""

    def _parse_llm_output(self, response: str) -> List[Dict]:
        """解析LLM输出的JSON"""
        try:
            # 尝试提取JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if isinstance(data, list):
                    return data
            return []
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Failed to parse LLM output: %s", e)
            return []