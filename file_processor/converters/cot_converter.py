"""
COT (Chain-of-Thought) 数据集转换器
基于指令微调数据，补充 chain_of_thought 推理链
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional

from ..readers.base import ReaderResult
from ..utils.lang_detect import detect_language

logger = logging.getLogger(__name__)


class COTConverter:
    """转换为COT格式数据集"""

    def __init__(self, max_samples: int = 0, llm_generator=None):
        self.max_samples = max_samples
        self.llm_generator = llm_generator

    def convert(self, result: ReaderResult) -> List[Dict]:
        """将读取结果转换为COT样本列表"""
        if not result.success or not result.text:
            return []

        text = result.text.strip()
        samples = []

        # 1. 从结构化内容提取COT（步骤、流程、因果）
        step_samples = self._extract_step_cot(text, result)
        samples.extend(step_samples)

        # 2. 从定义/分类内容提取COT
        definition_samples = self._extract_definition_cot(text, result)
        samples.extend(definition_samples)

        # 3. 如果启用了LLM，用LLM增强
        if self.llm_generator and len(samples) < 10:
            llm_samples = self._generate_cot_with_llm(text, result)
            samples.extend(llm_samples)

        if self.max_samples > 0:
            samples = samples[:self.max_samples]

        return samples

    def _extract_step_cot(self, text: str, result: ReaderResult) -> List[Dict]:
        """从步骤/流程类文本提取COT"""
        samples = []
        source = os.path.basename(result.filepath)
        language = detect_language(text)

        # 匹配编号步骤： 1. xxx \n 2. xxx \n 3. xxx
        step_patterns = [
            re.findall(r'(?:^|\n)\s*(\d+)[.、．\s]\s*(.+?)(?=\n\s*\d+[.、．]|\n\n|\Z)', text, re.DOTALL),
            re.findall(r'(?:^|\n)\s*([第一步-第十步]+)[：:]?\s*(.+?)(?=\n\s*[第一步-第十步]+|\n\n|\Z)', text, re.DOTALL),
            re.findall(r'(?:^|\n)\s*([阶段步骤环节][一二三四五六七八九十]+)[：:]?\s*(.+?)(?=\n\s*[阶段步骤环节][一二三四五六七八九十]+|\n\n|\Z)', text, re.DOTALL),
        ]

        for steps in step_patterns:
            if len(steps) >= 2:
                step_texts = [f"{num}. {desc.strip()}" for num, desc in steps]
                instruction_text = self._find_step_context(text, steps[0][0])
                if len(instruction_text) > 50:
                    instruction_text = instruction_text[:50] + "..."

                # 生成 "步骤是什么" 类指令
                cot_steps = "\n".join(step_texts)
                conclusion = step_texts[-1] if step_texts else "完成"

                samples.append({
                    "instruction": f"请逐步说明{instruction_text}的完整过程。",
                    "input": "",
                    "chain_of_thought": cot_steps,
                    "output": conclusion.split(". ", 1)[-1] if ". " in conclusion else conclusion,
                    "source": source,
                })

                # 如果步骤多，生成"总共几步"类指令
                if len(steps) >= 3:
                    samples.append({
                        "instruction": f"实现{instruction_text}需要哪些步骤？请分步说明。",
                        "input": "",
                        "chain_of_thought": f"总共需要{len(steps)}个步骤：\n" + cot_steps,
                        "output": f"实现{instruction_text}需要{len(steps)}个步骤。",
                        "source": source,
                    })

                if len(samples) >= 10:
                    return samples

        return samples

    def _find_step_context(self, text: str, first_step: str) -> str:
        """查找步骤的上下文标题"""
        idx = text.find(first_step)
        if idx < 0:
            return "该任务"
        before = text[:idx]
        headings = re.findall(r'^#{1,3}\s+(.+)$', before, re.MULTILINE)
        if headings:
            return headings[-1].strip()
        return "该任务"

    def _extract_definition_cot(self, text: str, result: ReaderResult) -> List[Dict]:
        """从定义/分类内容提取COT"""
        samples = []
        source = os.path.basename(result.filepath)

        # 匹配"因为...所以...", "如果...那么...", "首先...其次...最后..."
        patterns = [
            (r'因为(.+?)，所以(.+?)[。.]', "因果关系推理"),
            (r'如果(.+?)，(?:那么|就|则)(.+?)[。.]', "条件推理"),
            (r'首先[,，](.+?)[,，]其次[,，](.+?)[,，]最后[,，](.+?)[。.]', "三步推理"),
        ]

        for pattern, logic_type in patterns:
            for m in re.finditer(pattern, text):
                groups = m.groups()
                if len(groups) == 2:
                    premise, conclusion = groups
                    samples.append({
                        "instruction": f"根据已知信息，分析{premise}会得出什么结论？请给出推理过程。",
                        "input": "",
                        "chain_of_thought": f"推理过程：\n1. 已知条件：{premise.strip()}\n"
                                              f"2. 逻辑分析：根据{logic_type}原则\n"
                                              f"3. 推理结论：{conclusion.strip()}",
                        "output": conclusion.strip(),
                        "source": source,
                    })
                elif len(groups) == 3:
                    first, second, third = groups
                    samples.append({
                        "instruction": f"请逐步分析完整过程。",
                        "input": "",
                        "chain_of_thought": f"第一步骤：{first.strip()}\n"
                                              f"第二步骤：{second.strip()}\n"
                                              f"第三步骤：{third.strip()}",
                        "output": third.strip(),
                        "source": source,
                    })

                if len(samples) >= 10:
                    return samples

        return samples

    def _generate_cot_with_llm(self, text: str, result: ReaderResult) -> List[Dict]:
        """用LLM生成COT数据"""
        if not self.llm_generator:
            return []

        samples = []
        source = os.path.basename(result.filepath)
        language = detect_language(text)

        # 先看有没有instruction step中生成的指令，基于那些生成COT
        from .instruction_converter import InstructionConverter
        ic = InstructionConverter(max_samples=5)
        instructions = ic._generate_from_templates(text, result)

        for inst in instructions[:3]:
            try:
                prompt = self._build_cot_prompt(
                    inst["instruction"],
                    inst["output"],
                    text,
                    language
                )
                response = self.llm_generator.generate(
                    prompt,
                    system_prompt="你是一个专业的推理链生成助手。请为问答对补充详细的推理步骤。"
                )

                cot = self._parse_cot_output(response)
                if cot:
                    samples.append({
                        "instruction": inst["instruction"],
                        "input": inst.get("input", ""),
                        "chain_of_thought": cot,
                        "output": inst["output"],
                        "source": source,
                    })
            except Exception as e:
                logger.warning("LLM COT generation failed: %s", e)
                continue

        return samples

    def _build_cot_prompt(self, instruction: str, output: str,
                          context: str, language: str) -> str:
        """构建COT生成prompt"""
        if language == "zh":
            return f"""请根据以下问答对，生成详细的推理过程(chain-of-thought)。

原始文本的上下文（供参考）：
{context[:800]}

问答对：
指令：{instruction}
回答：{output}

要求：
1. 推理过程要逻辑清晰，分步骤展开
2. 每个步骤都有理有据
3. 最终结论和给出的回答一致

请按以下格式输出（只输出推理过程本身，不要额外解释）：
1. ...
2. ...
3. ...
因此结论是：{output}"""
        else:
            return f"""Generate a detailed chain-of-thought reasoning process for the following Q&A pair.

Context:
{context[:800]}

Q&A:
Instruction: {instruction}
Answer: {output}

Requirements:
1. Clear logical reasoning, step by step
2. Each step must be justified
3. Final conclusion must match the given answer

Output format (reasoning process only):
1. ...
2. ...
3. ...
Therefore, the answer is: {output}"""

    def _parse_cot_output(self, response: str) -> Optional[str]:
        """解析COT输出"""
        response = response.strip()
        if len(response) < 10:
            return None
        return response