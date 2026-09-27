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

    def __init__(self, max_samples: int = 0, llm_generator=None,
                 sample_check_ratio: float = 0.0,
                 review_output: str = "",
                 verify_code_exec: bool = False,
                 llm_max_attempts: int = 1):
        self.max_samples = max_samples
        self.llm_generator = llm_generator
        # 样本级校验（转换后）
        self.sample_check_ratio = sample_check_ratio   # LLM 语义抽检比例（0=关）
        self.review_output = review_output             # 争议样本导出路径（人工双人审核）
        self.verify_code_exec = verify_code_exec       # 代码题运行验证（默认关）
        self.llm_max_attempts = llm_max_attempts       # 拒绝采样: chunk 生成硬校验全废时重试上限

    def convert(self, result: ReaderResult) -> List[Dict]:
        """将读取结果转换为指令样本列表"""
        if not result.success or not result.text:
            return []

        text = result.text.strip()
        # 剥离扫描/转换残留的 markdown 加粗标记（保留内容，去除 ** 对）
        text = re.sub(r'\*\*([^*\n]+?)\*\*', r'\1', text)
        samples = []

        # 先尝试用模板规则生成（加帽：模板样本封顶 10 条，保证 LLM 通道有补充空间；
        # 否则长文档模板先吃满 max_samples，LLM 增强形同虚设）
        template_samples = self._generate_from_templates(text, result)
        samples.extend(template_samples[:10])

        # 如果启用了LLM，用LLM补充更多样化的指令
        if self.llm_generator and len(samples) < 20:
            llm_samples = self._generate_with_llm(text, result)
            samples.extend(llm_samples)

        # 限制数量
        if self.max_samples > 0:
            samples = samples[:self.max_samples]

        # 确定性硬校验（隐私红线 / 代码运行）: 不合格直接剔除
        samples = self._hard_validate(samples)

        # LLM 语义抽检(答案基于材料/指令可执行/无幻觉; 争议样本导出人工审核)
        if self.sample_check_ratio > 0 and self.llm_generator and samples:
            samples = self._semantic_sample_check(samples)

        return samples

    def _hard_validate(self, samples: List[Dict]) -> List[Dict]:
        """确定性硬校验（《数据集介绍.md》6.2/1.3）：隐私红线、代码运行。
        不合格样本直接剔除（机器可复现的客观判定）。"""
        from ..utils.validation import privacy_scan, code_exec_check
        bad_reasons = {}
        for i, s in enumerate(samples):
            hits = privacy_scan((s.get("input") or "") + " " + (s.get("output") or ""))
            if hits:
                bad_reasons[i] = f"隐私未脱敏({','.join(sorted(hits))})"
                continue
            if self.verify_code_exec:
                r = code_exec_check((s.get("input") or "") + " " + (s.get("output") or ""))
                if r is False:
                    bad_reasons[i] = "代码运行失败或输出不符"
                    continue
        if bad_reasons:
            counts = {}
            for r in bad_reasons.values():
                counts[r] = counts.get(r, 0) + 1
            logger.warning("SFT 确定性校验剔除 %d 条: %s", len(bad_reasons), counts)
            samples = [s for i, s in enumerate(samples) if i not in bad_reasons]
        return samples

    # 开卷样本材料/输出长度上限
    MATERIAL_MAX = 1200
    OUTPUT_MAX = 1000

    # 每份来源的产条上限: 由体量决定(每 ~1500 字 1 条), 夹在 [3, 40]
    # 口径: 有多少字就分析多少字, 能生成多少就生成多少 —— 不再固定"只吃前 5 块"
    _SFT_CHARS_PER_SAMPLE = 1500
    _SFT_MIN_PER_SOURCE = 3
    _SFT_MAX_PER_SOURCE = 40

    def _source_budget(self, text: str) -> int:
        """每份来源的目标条数(按体量), 夹在 [_SFT_MIN_PER_SOURCE, _SFT_MAX_PER_SOURCE]"""
        n = int(round(len(text or "") / float(self._SFT_CHARS_PER_SAMPLE)))
        return max(self._SFT_MIN_PER_SOURCE, min(self._SFT_MAX_PER_SOURCE, n))

    def _generate_from_templates(self, text: str, result: ReaderResult) -> List[Dict]:
        """从文本结构中用模板生成指令对

        字段语义约束（训练用途）：
        - 答案内容来自某篇具体文档的样本（标题阅读、列表提取），来源材料
          必须放进 input——模型学的是"根据所给材料作答"（开卷），否则等于
          要求模型闭卷背诵一篇没见过的文档，是在教幻觉；
        - 自包含通识样本（定义句抽取：题目不含特定文档指向）input 为空是合规的。
        """
        samples = []
        source = os.path.basename(result.filepath)

        # 1. 标题段落 → 阅读理解（旧实现的标题模板 output 恒为 None，从未生效；
        #    现按"材料进 input、抽取式要点进 output"落地）
        heading_pattern = re.compile(r'^#{1,3}\s+(.+)$', re.MULTILINE)
        headings = list(heading_pattern.finditer(text))

        for i, m in enumerate(headings):
            title = m.group(1).strip()
            start = m.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            content = text[start:end].strip()
            if len(content) < 60:
                continue
            material = content[:self.MATERIAL_MAX]
            # 要点必须从截断后的材料里抽取：output 引用 input 之外的内容=开卷变闭卷
            key_points = self._key_points(material)
            # 纯编号标题（第四条/（三）/附录2）不携带语义，指代进题面反成噪声
            generic = bool(re.fullmatch(
                r'第[一二三四五六七八九十0-9]+[章节条款项]|附录\s*\d*|[（(]?[一二三四五六七八九十]{1,3}[）)]?',
                title))
            if contains_english(title) and not contains_chinese(title):
                samples.append({
                    "instruction": f"Based on the following material about \"{title}\", "
                                   f"summarize its key points.",
                    "input": material,
                    "output": key_points,
                    "source": source,
                })
            else:
                samples.append({
                    "instruction": (f"根据以下关于「{title}」的材料，总结其主要内容的要点。"
                                    if not generic else
                                    "根据以下材料，总结其主要内容的要点。"),
                    "input": material,
                    "output": key_points,
                    "source": source,
                })
            if len(samples) >= 30:
                return samples

        # 2. 列表 → 提取整理（材料=列表原行，输出=编号整理结果）
        list_pattern = re.compile(r'^[\-\*]\s+(.+)$', re.MULTILINE)
        list_items = [item.strip() for item in list_pattern.findall(text) if item.strip()]

        if len(list_items) >= 3:
            context = self._find_list_context(text, list_items)
            # 预算内截断，保证输出条目全部见于材料
            kept, used = [], 0
            for item in list_items:
                if used + len(item) + 1 > self.MATERIAL_MAX:
                    break
                kept.append(item)
                used += len(item) + 1
            if len(kept) >= 3:
                numbered = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(kept))
                ref = f"其中关于「{context}」的" if context else "其中的"
                samples.append({
                    "instruction": f"根据以下材料，将{ref}条目逐条列举并编号。",
                    "input": "\n".join(kept),
                    "output": numbered[:2000],
                    "source": source,
                })
            if len(samples) >= 30:
                return samples

        # 2b. 编号条目块（政策白皮书"一、二、三…"措施列表）→ 列举阅读。
        #     并行措施不是时序流程（COT 收它会产出"逐步说明14点倡议的完整过程"
        #     这类语义错配样本），正确任务形态是"根据材料逐条列举"。
        numbered_re = re.compile(
            r'^[（(]?[一二三四五六七八九十0-9]{1,4}[）)、.．][ 　]*(\S.*)$', re.MULTILINE)
        ms = list(numbered_re.finditer(text))
        i = 0
        while i < len(ms):
            j = i
            # 相邻条目之间只允许空白（保证是同一个列举块）
            while j + 1 < len(ms) and not text[ms[j].end():ms[j + 1].start()].strip():
                j += 1
            block = ms[i:j + 1]
            i = j + 1
            descs = [b.group(1).strip() for b in block]
            if len(descs) < 3:
                continue
            lens = sorted(len(d) for d in descs)
            if lens[len(lens) // 2] < 10:
                continue  # 短词占多数：目录/章节名，不是实质条目
            # 截到材料预算内，保证"答案不超出所给材料"
            budget, kept, kept_segs = self.MATERIAL_MAX, [], []
            raw = ""
            for b, d in zip(block, descs):
                seg = text[b.start():b.end()]
                if len(raw) + len(seg) > budget:
                    break
                raw += seg + "\n"
                kept.append(d)
                kept_segs.append(seg)
            if len(kept) < 3:
                continue
            context = self._find_list_context(text, [kept[0]])
            if context and len(context) > 30:
                context = ""
            ref = f"其中关于「{context}」的" if context else "其中的"
            # output 保留原文条目行(含(一)(二)等层级编号标签), 与 input 逐条对应;
            # 重编号会剥掉"(二)"标签造成 input/output 不一致, 被审查判 fail
            out_numbered = "\n".join(s.strip() for s in kept_segs)
            samples.append({
                "instruction": f"根据以下材料，将{ref}内容要点逐条列举并编号。",
                "input": raw.strip()[:self.MATERIAL_MAX],
                "output": out_numbered[:2000],
                "source": source,
            })
            if len(samples) >= 30:
                return samples

        # 3. 定义式句子 → 通识问答（自包含，input 允许为空）。严格清洗：
        #    旧正则 (.{10,200}) 会把后文整段吞进答案、编号前缀留在术语里，
        #    产出"什么是2. 我校…方式：一？"这类碎片——训练用途不可接受。
        def_pattern = re.compile(
            r'^(.{2,30})(是指|指的是|即|是(?:一[种类])?|属于|意为)(.{5,200})', re.MULTILINE)
        for m in def_pattern.finditer(text):
            sample = self._definition_sample(m.group(1), m.group(2), m.group(3), source)
            if sample:
                samples.append(sample)
                if len(samples) >= 30:
                    return samples

        return samples

    _TERM_MAX = 16
    _TERM_BAD_TAIL = re.compile(r'(仍然|仍|并|都|也|将|应当|必须|可以|已|尚未|正在|需要|能够)$')

    def _clean_term(self, term: str) -> str:
        """剥离 markdown 加粗残留、列表/章节编号前缀与首尾杂标点"""
        term = term.replace('*', '').strip().strip('，,、；;：:。 ').strip()
        prev = None
        while prev != term:
            prev = term
            term = re.sub(r'^[（(][一二三四五六七八九十0-9]{1,3}[）)]\s*', '', term)
            term = re.sub(r'^第[一二三四五六七八九十0-9]+[章节条款项][ 　]*', '', term)
            term = re.sub(r'^[一二三四五六七八九十0-9]{1,3}[、.．:：]\s*', '', term)
            term = term.strip()
        return term

    def _definition_sample(self, term: str, connector: str, rest: str,
                           source: str) -> Optional[Dict]:
        """构造通过清洗门控的定义式样本，不合格返回 None"""
        term = self._clean_term(term)
        if not (2 <= len(term) <= self._TERM_MAX):
            return None
        if re.search(r'[：:]', term) or self._TERM_BAD_TAIL.search(term):
            return None
        definition = rest.strip()
        if not definition or '是指' in definition[:12]:
            return None
        # 截到首个句末标点：答案自包含且不吃进下一条编号
        end = re.search(r'[。！？!?]', definition)
        if not end:
            return None
        definition = definition[:end.end()]
        if len(definition) < 8:
            return None
        comma = "，" if connector.startswith(("是指", "指的是")) else ""
        return {
            "instruction": f"什么是{term}？",
            "input": "",
            "output": f"{term}{comma}{connector}{definition}",
            "source": source,
        }

    def _key_points(self, content: str, max_points: int = 5) -> str:
        """每段首句组成编号要点（确定性抽取，避免 output 与 input 整段重复）"""
        points = []
        for para in content.split("\n\n"):
            para = para.strip()
            if len(para) < 15:
                continue
            cut = None
            for punct in "。！？!?.":
                idx = para.find(punct, 12)
                if 0 < idx <= 160 and (cut is None or idx < cut):
                    cut = idx + 1
            sent = (para[:cut] if cut else para[:120]).strip()
            points.append(f"{len(points) + 1}. {sent}")
            if len(points) >= max_points:
                break
        return "\n".join(points) if points else content[:200].strip()

    def _find_list_context(self, text: str, items: List[str]) -> str:
        """查找列表项的上下文（前面最近的标题或句子）；找不到返回空串"""
        if not items:
            return ""

        first_item = items[0]
        idx = text.find(first_item)
        if idx < 0:
            return ""

        # 找前面的标题
        before = text[:idx]
        heading_m = re.findall(r'^#{1,3}\s+(.+)$', before, re.MULTILINE)
        if heading_m:
            return heading_m[-1].strip()

        # 找前面第一个句号/冒号前的文本
        sentences = re.split(r'[。：:]', before)
        if sentences:
            ctx = re.sub(r'\s+', ' ', sentences[-1].strip())
            # 去掉条目自身编号粘连（"…政策宣示 一、"）
            ctx = re.sub(r'[（(]?[一二三四五六七八九十0-9]{1,4}[）)、.．]\s*$', '', ctx)
            ctx = ctx.strip(' ，,、；;。')
            if len(ctx) > 2:
                return ctx

        return ""

    def _generate_with_llm(self, text: str, result: ReaderResult) -> List[Dict]:
        """调用LLM生成指令-输出对"""
        if not self.llm_generator:
            return []

        samples = []
        source = os.path.basename(result.filepath)
        language = detect_language(text)

        # 按长度切块, 逐块生成指令 —— 块数按**体量**, 不再固定只处理前 5 块
        chunks = self._split_into_chunks(text, 1500)
        budget = self._source_budget(text)
        for chunk in chunks:
            if len(samples) >= budget:
                logger.info("SFT 该来源条数达上限 %d(按体量), 停止后续分块", budget)
                break
            # 拒绝采样(失败重试制): 该块一次生成产出 0 条有效样本才重试
            # (上限 llm_max_attempts); 有 1 条即停 → 多数场景零额外成本
            chunk_samples: List[Dict] = []
            for attempt in range(max(1, self.llm_max_attempts)):
                try:
                    prompt = self._build_instruction_prompt(chunk, language)
                    response = self.llm_generator.generate(
                        prompt,
                        system_prompt="你是一个专业的数据标注助手。请根据文本内容生成高质量的指令-响应对。"
                    )

                    parsed = self._parse_llm_output(response)
                    from ..utils.validation import trim_incomplete_output
                    for item in parsed:
                        # 兜底: 指代材料的题(input 为空)自动补材料节选,
                        # 防止"该文件/该方案"类指令无依据输出
                        inp = item.get("input", "")
                        if not inp and self._REFER_RE.search(item.get("instruction", "")):
                            item["input"] = chunk[:600]
                        # input 节选清理: 尾部残缺编号行("1." 行首且无句号)裁掉,
                        # 尾部非完整句裁到最近句号——审查员据此判"材料截断不完整"
                        item["input"] = self._tidy_input(item.get("input", ""))
                        # 截断兜底: max_tokens 腰斩的 output 裁到最近完整句;
                        # 裁后不足 min_len 或全无标点 → 视为废样本丢弃
                        out = trim_incomplete_output(item.get("output", ""))
                        if out is None:
                            logger.warning("SFT LLM 样本输出被截断且无有效完整句, 丢弃: %s",
                                           str(item.get("output"))[:40])
                            continue
                        item["output"] = out
                        # 数字一致性硬查(防幻觉): output 中的数值须能追溯到 input/上下文
                        material = item.get("input", "") or chunk
                        from ..utils.validation import numeric_consistency
                        if not numeric_consistency(item.get("output", ""), material,
                                       full_material=text):
                            logger.warning("SFT LLM 样本未过数字一致性校验, 丢弃: %s",
                                           str(item.get("output"))[:40])
                            continue
                        item["source"] = source
                        chunk_samples.append(item)

                    if attempt > 0:
                        logger.info("SFT LLM chunk 重试第 %d 次: 产出 %d 条有效样本",
                                    attempt + 1, len(chunk_samples))
                except Exception as e:
                    logger.warning("LLM generation failed: %s", e)
                if chunk_samples:
                    break
            samples.extend(chunk_samples)

        return samples

    _REFER_RE = re.compile(r'该文本|该文件|该方案|该材料|上述|其中|这段文字|以上材料|以下材料|文中|文本中|材料中|原文|所给材料|内容中|文章中|文件信息|根据材料|基于材料|材料，|材料。')
    _NUM_LINE_RE = re.compile(r'^\s*(?:\d+|[一二三四五六七八九十]+|（?(?:第)?[一二三四五六七八九十0-9]{1,4}）?)[.、．）]')

    @staticmethod
    def _tidy_input(inp: str) -> str:
        """LLM 填充的 input 节选清理：从尾部逐行裁掉残缺行
        (编号行首且无句号 / 末尾行不以句号结尾)，直到末行完整
        (以句号结尾)或为空——自包含题允许空 input。"""
        if not inp:
            return ""
        lines = [l for l in inp.rstrip().splitlines() if l.strip()]
        while lines:
            last = lines[-1].strip()
            if last.endswith('。'):
                break
            if InstructionConverter._NUM_LINE_RE.match(last):
                lines.pop()
                continue
            break
        while lines and not lines[-1].strip().endswith('。'):
            lines.pop()
        out = "\n".join(lines).strip()[:2000]
        # 保护: 输入非空但清理后为空(如导航区全是无句号短行)时保留末行,
        # 否则兜底补的材料会被自身清理撤销, 指代题又变 input 空
        return out if out else (inp.strip()[:500] if inp else "")

    def _split_into_chunks(self, text: str, max_chars: int) -> List[str]:
        """行级分块: 以行为单位累积(行=完整段落/编号条目), 超预算 flush。

        旧实现按 \\n\\n 段落切——政策正文("1.xxx\\n2.xxx" 单换行无空行)会被
        当成一个长段, 产生"导航区 114 字符碎片"或 8000 字符巨块, LLM 无法
        基于碎片/巨块生成有效样本。行级保证编号条目不拆断。
        """
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        chunks, cur, used = [], [], 0
        for ln in lines:
            if used + len(ln) + 1 > max_chars and cur:
                chunks.append('\n'.join(cur))
                cur, used = [], 0
            # 兜底: 单行超长(清洗后无换行的长文)按句切
            while len(ln) > max_chars:
                cut = -1
                for mLen in (40, 20):
                    pos = max(ln.rfind(c, 0, max_chars) for c in '。；;') if any(c in ln[:max_chars] for c in '。；;') else -1
                    if pos >= 0:
                        cut = pos + 1
                        break
                if cut <= 0:
                    cut = max_chars
                if cur:
                    chunks.append('\n'.join(cur))
                    cur, used = [], 0
                chunks.append(ln[:cut])
                ln = ln[cut:]
            cur.append(ln)
            used += len(ln) + 1
        if cur:
            chunks.append('\n'.join(cur))
        return chunks or ([text] if text else [])

    def _build_instruction_prompt(self, text: str, language: str) -> str:
        """构建LLM生成指令的prompt（单模板参数化 + 真实数据集 few-shot 风格参考）"""
        if language == "zh":
            rules = (
                "要求：\n"
                "1. 指令要多样，覆盖不同信息维度：总结、提取、问答（开卷+自包含)、改写、分类、推理、建议、比较\n"
                "2. 同一段文本至少覆盖 3-5 种不同形态，不要重复类似提问\n"
                "3. 答案必须严格基于文本内容，不编造\n"
                "4. 题目需要参考原文材料才能作答的，把相关材料节选放入 input；题目自包含（不指向特定文档的通识问答）input 留空\n"
                "4.1 凡指令中出现「该文本/该文件/该方案/该材料/上述/其中/这段文字」等指代词，都是引用材料的题目，input 必须非空（放入材料节选）；\n"
                "    材料中没有答案的问题（如出处/URL/作者信息）不要生成；仅当材料足够回答时才生成该条\n"
                "5. 输出JSON数组格式，每个对象**必须且只能**包含三个字段：instruction、input、output（不要用 response 等其他字段名）\n"
                "6. 列举/清点类答案要点化：每条一行、必须是一句成分完整的短句（有主语、以句号结尾），每条不超过35字；"
                "严禁输出无主语的短语残片（如“比例不得低于20%”是残片，应写“非化石能源消费占比不得低于20%左右。”）"
            )
            shots = (
                '{"instruction": "三原色是什么？", "input": "", "output": "三原色通常指的是红色、绿色和蓝色（RGB）。……在印刷和绘画中，三原色指的是以颜料为基础的红、黄和蓝颜色（RYB）。……三原色的具体定义并不唯一，不同的颜色系统可能会采用不同的三原色。"}'
                '  ← shibing624/alpaca-zh 真实样本(原文) · 自包含通识问答，input 留空\n'
                '{"instruction": "根据以下材料，总结其主要内容的要点。", "input": "（材料节选……）", "output": "1. ……\\n2. ……"}'
                '  ← 开卷问答，材料节选必须放进 input'
            )
        else:
            rules = (
                "Requirements:\n"
                "1. Diverse instructions covering multiple dimensions: summarization, extraction, "
                "open-book & self-contained Q&A, rewriting, classification, reasoning, advice, comparison\n"
                "2. Cover at least 3-5 different task types per text chunk; do not repeat similar questions\n"
                "3. Answers must be strictly based on the text, no fabrication\n"
                "4. If a question requires the source material to answer, put an excerpt in \"input\"; "
                "leave \"input\" empty only for self-contained general-knowledge questions\n"
                "5. Output a JSON array; every object MUST contain exactly instruction, input, output "
                "(do not use other field names like response)\n"
                "6. For listing/enumeration answers put each item on its own line as a COMPLETE short "
                "sentence (with subject, ending in a period), ≤35 chars per line; "
                "never output subject-less fragments (e.g. write \"The share target is no less than 20%.\" "
                "instead of \"no less than 20%\")"
            )
            shots = (
                '{"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \\n2. Exercise regularly to keep your body active and strong. \\n3. Get enough sleep and maintain a consistent sleep schedule."}'
                '  <- tatsu-lab/alpaca real sample (verbatim) · self-contained Q&A, input empty\n'
                '{"instruction": "Based on the following material, summarize its key points.", "input": "(material excerpt)", "output": "1. ...\\n2. ..."}'
                '  <- open-book Q&A, excerpt goes into input'
            )
        return f"""从以下文本中生成3-5个高质量的指令-响应对。

{rules}

参考真实数据集的样本风格（保持这样的字段结构和简洁度，不要照抄题目）：
{shots}

文本内容：
{text}

请按以下格式输出（严格JSON数组，不要包含其他文字）：
[
  {{"instruction": "...", "input": "...", "output": "..."}},
  ...
]"""

    def _parse_llm_output(self, response: str) -> List[Dict]:
        """解析LLM输出的JSON,并归一化字段(容错 response/output 混用)"""
        try:
            # 尝试提取JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if isinstance(data, list):
                    normalized = []
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        # 模型偶尔把 answer 字段写成 response -> 归一化
                        if "response" in item and "output" not in item:
                            item["output"] = item.pop("response")
                        item.setdefault("input", "")
                        # 缺 instruction 或 output 的样本没有训练价值,丢弃
                        if item.get("instruction") and item.get("output"):
                            normalized.append(item)
                    return normalized
            return []
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Failed to parse LLM output: %s", e)
            return []

    # ============ 样本级 LLM 语义抽检(答案基于材料/指令可执行/无幻觉/无有害) ============
    _CHECK_PROMPT = """请从严审查以下指令微调训练样本。四项检查：
1. output 是否严格基于 input 材料作答(自包含通识题须合理正确), 无材料外事实/数字, 无幻觉
2. instruction 是否明确、可执行
3. input 语义是否正确(开卷题含材料节选, 自包含题为空)
4. 无有害/违法/危险内容, 无隐私信息

输出严格 JSON: {{"pass": true 或 false, "reason": "一句话理由"}}

样本(JSON):
{data}
"""

    def _semantic_sample_check(self, samples: List[Dict]) -> List[Dict]:
        """LLM 语义抽检：抽样审查。明确 pass=false 或无法判定的样本**不移除主集**，
        而是标记后写入待人工审核清单（双人审核/仲裁为人工流程，见《数据集介绍.md》1.2）。"""
        if not self.llm_generator:
            return samples
        import random
        n = min(len(samples), max(1, int(len(samples) * self.sample_check_ratio)))
        idx = random.sample(range(len(samples)), n)
        review, review_idx = [], set()  # (原样本, 原因) 与对应下标
        for i in idx:
            s = samples[i]
            try:
                resp = self.llm_generator.generate(
                    self._CHECK_PROMPT.format(data=json.dumps({
                        "instruction": s.get("instruction", ""),
                        "input": s.get("input") or "",
                        "output": s.get("output") or "",
                    }, ensure_ascii=False)),
                    system_prompt="你是训练数据质量审查员，严格把关，宁可剔除也不放过坏样本。")
                m = re.search(r'"pass"\s*:\s*(true|false)', resp, re.IGNORECASE)
                if m and m.group(1).lower() == 'true':
                    continue  # 判定合格
                if m:
                    rm = re.search(r'"reason"\s*:\s*"([^"]*)"', resp)
                    review.append((s, 'LLM 判定不合格: ' + (rm.group(1) if rm else '未给出理由')))
                else:
                    review.append((s, 'LLM 未返回可解析判定(争议样本)'))
                review_idx.add(i)
            except Exception as e:
                review.append((s, f'抽检调用异常: {e}'))
                review_idx.add(i)
        if review:
            # 争议样本移出主集(不进训练), 导出人工审核; 人工仲裁后可决定救回/丢弃
            logger.warning("LLM 抽检移除 %d/%d 个争议样本并导出人工审核 (总 %d 条)",
                           len(review), n, len(samples))
            self._flush_review(review)
            samples = [s for k, s in enumerate(samples) if k not in review_idx]
        return samples

    def _flush_review(self, review):
        """争议样本落盘成待人工审核清单（review_output 配置时）；空路径则仅记日志。"""
        if not self.review_output or not review:
            return
        import time
        from ..utils.locks import REVIEW_LOCK
        try:
            with REVIEW_LOCK, open(self.review_output, 'a', encoding='utf-8') as f:
                for s, reason in review:
                    rec = dict(s)
                    rec['_review_reason'] = reason
                    rec['_review_ts'] = time.strftime('%Y-%m-%d %H:%M:%S')
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            logger.info("待人工审核样本 %d 条已追加: %s", len(review), self.review_output)
        except OSError as e:
            logger.warning("无法写入审核清单 %s: %s", self.review_output, e)