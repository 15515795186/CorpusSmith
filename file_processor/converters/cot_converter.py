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
        # 剥离扫描/转换残留的 markdown 加粗标记（保留内容，去除 ** 对）
        text = re.sub(r'\*\*([^*\n]+?)\*\*', r'\1', text)
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

        # 匹配编号步骤。注意：仅靠"数字+分隔符"不够——arXiv 论文的章节编号
        # （1. Introduction / 2. Related Work）会被当成"40 步流程"。
        # 守卫：步数 2~12、描述中位长 ≥10 字（真实步骤是句子，章节名是短语）。
        step_patterns = [
            re.findall(r'(?:^|\n)\s*(\d+)[.、．]\s*(.+?)(?=\n\s*\d+[.、．]|\n\n|\Z)', text, re.DOTALL),
            re.findall(r'(?:^|\n)\s*(第[一二三四五六七八九十]+步)[：:]?\s*(.+?)(?=\n\s*第[一二三四五六七八九十]+步|\n\n|\Z)', text, re.DOTALL),
            re.findall(r'(?:^|\n)\s*((?:阶段|步骤|环节)[一二三四五六七八九十]+)[：:]?\s*(.+?)(?=\n\s*(?:阶段|步骤|环节)[一二三四五六七八九十]+|\n\n|\Z)', text, re.DOTALL),
        ]

        # 两种样本形态，语义不同：
        # ① 流程逐步说明——仅当有强时序证据（步骤文本自身含顺序词，或紧邻
        #    引导句含流程词）；对"1. 公司甲 2. 公司乙"式清单硬说"完整流程"
        #    就是语义错配，宁缺毋滥。
        # ② 条目清点——任何通过守卫的编号块都成立（数数不预设时序）。
        step_order_hint = re.compile(r'首先|然后|接着|随后|此后|最后|第一步|第二步|第三步')
        window_hint = re.compile(r'流程|过程如下|步骤如下|程序如下|操作步骤|办理程序|按下列程序')
        for cand in step_patterns:
            steps = COTConverter._real_process_steps(cand)
            if steps:
                # 步骤文本装进材料预算：计数答案必须与 input 里可见的条目一致，
                # 超预算的条目连同其计数一并舍弃（答案超出材料=教幻觉）
                kept, used = [], 0
                for num, desc in steps:
                    line = f"{num}. {re.sub(r'\s+', ' ', desc.strip())}"
                    if used + len(line) + 1 > 1500:
                        break
                    kept.append(line)
                    used += len(line) + 1
                if len(kept) < 2:
                    continue
                instruction_text = self._find_step_context(text, steps[0][0])
                if len(instruction_text) > 50:
                    instruction_text = instruction_text[:50] + "..."

                cot_steps = "\n".join(kept)
                material = cot_steps
                conclusion = kept[-1]
                topic = f"「{instruction_text}」" if instruction_text else ""

                first_pos = text.find((steps[0][1] or "")[:15])
                window = text[max(0, first_pos - 80):first_pos] if first_pos >= 0 else ""
                is_procedure = (sum(bool(step_order_hint.search(d)) for _, d in steps) >= 2
                                or bool(window_hint.search(window)))
                if is_procedure:
                    samples.append({
                        "instruction": f"根据以下材料，逐步说明{topic}完整过程。" if topic else
                                       "根据以下材料，逐步说明其中描述的完整流程。",
                        "input": material,
                        "chain_of_thought": cot_steps,
                        "output": conclusion.split(". ", 1)[-1] if ". " in conclusion else conclusion,
                        "source": source,
                    })

                samples.append({
                    "instruction": f"根据以下材料，清点{topic}总共列出了几个条目？请逐条核对。" if topic else
                                   "根据以下材料，清点其中总共列出了几个条目？请逐条核对。",
                    "input": material,
                    "chain_of_thought": f"逐条清点材料：\n{cot_steps}\n"
                                        f"核对编号后共{len(kept)}个条目。",
                    "output": f"{topic}共列出{len(kept)}个条目。" if topic else
                              f"材料共列出{len(kept)}个条目。",
                    "source": source,
                })

                if len(samples) >= 10:
                    return samples

        return samples

    _SECTION_WORDS = re.compile(
        r'^(introduction|related\s+work|background|method(s|ology)?|approach|'
        r'experiments?|results?|evaluation|discussion|conclusions?|references?|'
        r'acknowledg(e)?ments?|abstract)\.?$', re.IGNORECASE)

    @staticmethod
    def _real_process_steps(steps):
        """校验并规整编号块，返回真实条目/流程步骤序列或 None。

        - 纯数字组按 +1 严格递增切段取最长段：合并编号（1,2,3,1,2,3…）
          和扫描版章节碎片（3./3./1./2.）都是目录噪声，不是同一个序列；
        - 步数 2~12、描述中位长度 ≥10 字：短词占多数的是章节名；
        - 英文论文节名（Introduction/Method…）过半即拒。"""
        if not steps:
            return None
        if all(re.fullmatch(r'\d+', n) for n, _ in steps):
            runs, cur = [], [steps[0]]
            for prev, nxt in zip(steps, steps[1:]):
                if int(nxt[0]) == int(prev[0]) + 1:
                    cur.append(nxt)
                else:
                    runs.append(cur)
                    cur = [nxt]
            runs.append(cur)
            steps = max(runs, key=len)
        if not (2 <= len(steps) <= 12):
            return None
        descs = [d.strip() for _, d in steps]
        lens = sorted(len(d) for d in descs)
        if lens[len(lens) // 2] < 10:
            return None
        titled = sum(1 for d in descs if COTConverter._SECTION_WORDS.match(d))
        if titled >= max(2, len(descs) // 2):
            return None
        return steps

    def _find_step_context(self, text: str, first_step: str) -> str:
        """查找步骤的上下文标题；找不到返回空串（由调用方改用无指代措辞）"""
        idx = text.find(first_step)
        if idx < 0:
            return ""
        before = text[:idx]
        headings = re.findall(r'^#{1,3}\s+(.+)$', before, re.MULTILINE)
        if headings:
            return headings[-1].strip()
        return ""

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
                    # 因果/条件类：已知条件已写进题面 → 题目自包含，input 空合规
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
                    # 旧题面"请逐步分析完整过程"没有对象；现在原句进 input，题面自包含
                    first, second, third = groups
                    samples.append({
                        "instruction": "根据以下材料，按顺序逐步分析其完整过程。",
                        "input": m.group(0)[:600].strip(),
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