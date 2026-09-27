"""
COT (Chain-of-Thought) 数据集转换器 —— **样本驱动**

《数据集介绍.md》4.x + skill `cot-dataset-generation` 的落地。整条链以**既有 SFT
问答对**为唯一素材(不得自造问答对), 按四个阶段走:

  [A] 判类   逐条 SFT 样本判四类(task_type)。三态输出:
             · 属于四类        -> 进候选池
             · 明确不属于四类  -> 不进 COT, 计数留证(总结/提取/改写/事实罗列等)
             · 难以区分        -> 争议样本导出人工审核, **不硬塞任何一类**
             判类只看"这条样本问的是什么", 判前剥掉模板套话与来源文件名;
             显式类别标记优先; "含编号/含数字"本身不构成推理 ——
             只有**存在可唯一验证答案**才算推理。
  [B] 搜据   用样本的 input/output 回原文检索, 取锚点窗口 + 数字补检索拼成 context
             (链只能引用 context 内内容, 从源头压幻觉)。
  [C] 生成   按类模板(few-shot 用 skill 的真实数据集样本原文)写 chain_of_thought;
             instruction/input/output **以 SFT 为基准**(措辞可微差, 核心本义不得变,
             禁止换题/换答案/新增事实); 校验不过按 llm_max_attempts 重试,
             重试后仍不过 -> **丢弃**。
  [D] 统计   四类占比只统计只报告(20/30/20/30 ±5%), **不裁剪、不改判、不删样本**
             (《数据集介绍.md》4.5 的口径是"整体分布满足", 差额归素材层解决)。

字段(规范 3.2/4.2/8.1/8.2): instruction/input/output/chain_of_thought/task_type/
verifiable/final_answer_correct/reasoning_correct/source; id 由写入层按写入序分配。
`final_answer_correct`、`reasoning_correct` **只在机器验证通过时填 true**, 未验证为 null。
`source` 记录"派生自哪条 SFT 样本 + 哪个语料"。
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from ..utils.lang_detect import detect_language

logger = logging.getLogger(__name__)


class COTConverter:
    """以既有 SFT 问答对为素材生成 COT 数据集"""

    #: 四类(《数据集介绍.md》4.5)
    CLASSES: Tuple[str, ...] = ("解释", "推理", "建议", "比较")
    #: 判类第三态: 信号冲突/不足 —— 不硬判, 导出人工审核
    UNCERTAIN = "难以区分"

    #: 四类目标占比(只作统计口径与告警基准, 不做裁剪)
    _DIST_TARGET = {"解释": 0.20, "推理": 0.30, "建议": 0.30, "比较": 0.20}

    def __init__(self, max_samples: int = 0, llm_generator=None,
                 sample_check_ratio: float = 0.0,
                 distribution_enforce=None,
                 distribution_tolerance: float = 0.05,
                 review_output: str = "",
                 verify_code_exec: bool = False,
                 llm_max_attempts: int = 1,
                 context_window: int = 1500):
        self.max_samples = max_samples
        self.llm_generator = llm_generator
        # 样本级 LLM 语义抽检(转换后, 从严; 争议样本移出主集)
        self.sample_check_ratio = sample_check_ratio
        # 兼容旧配置: 分布硬门禁已按规范改为"只统计不裁剪", 该开关不再裁剪样本
        self.distribution_enforce = distribution_enforce
        self.distribution_tolerance = distribution_tolerance
        self.review_output = review_output          # 争议样本导出路径(人工双人审核)
        self.verify_code_exec = verify_code_exec    # 代码题运行验证(默认关)
        self.llm_max_attempts = llm_max_attempts    # 拒绝采样: 校验不过的重试上限
        self.context_window = context_window        # 回原文搜据的窗口大小(字符)

        # 运行统计(供流水线汇报: 判类三态与丢弃原因都要留证)
        self.stats = {
            "sft_total": 0, "candidates": 0, "out_of_scope": 0,
            "uncertain": 0, "generated": 0, "dropped": 0,
            "dropped_reasons": {}, "by_class": {},
        }

    # ================= 【A】判类: SFT 样本 -> 四类 / 非四类 / 难以区分 =================
    # 信号一律按 skill 决策表实现, 中英同一套语义特征(英文形态此前缺失会让英文
    # 材料整类判空)。信号只作用于**判类素材** = 剥掉模板套话后的 instruction(+output),
    # 绝不整篇材料扫信号 —— 否则"材料里出现过'对比'两字"就会把总结类样本判成比较类。

    # 推理·算式信号: **只认真算式** —— 带等号 / 乘除 / 倍数。
    # 旧模式 "数字[+-*/]数字" 会把日期("2024-03-19")与数值区间("40-50%")当算术。
    _HINT_MATH = re.compile(
        r'\d+\s*[+\-×*/÷]\s*\d+\s*[=＝]|'
        r'\d+\s*[×*/÷]\s*\d+|'
        r'\d+\s*[加减乘除]\s*\d+\s*[等为得]|'
        r'[一二三四五六七八九十百千]+倍', re.IGNORECASE)
    # 计数提问: 问句要求清点/合计, 且材料可数 —— 存在唯一可验证答案才判推理
    _COUNT_CUE = re.compile(
        r'[共合计]计?\s*[一二三四五六七八九十百\d]+\s*[个项条种部章节]|'
        r'几(?:个|项|条|种|类|部分)|多少个|数量统计|条目数|总数|'
        r'\bhow many\b|\b(?:total|sum|altogether)\b|\bcount\b', re.IGNORECASE)
    # 推理·逻辑演绎信号: 只认条件式演绎; 单"因此/therefore"是结论词不算推理依据
    _HINT_REASON = re.compile(
        r'如果[^。\n]{0,20}(?:那么|就|则)|若[^。\n]{0,20}(?:则|就)|假设[^。\n]{0,20}(?:则|就)|'
        r'\bif\b[^.\\n]{0,60}\bthen\b|'
        r'由此(?:可见|可知)|推导(?:得出|可得)|(?:计算|求解|算出|求证|证明)', re.IGNORECASE)
    _HINT_LIST = re.compile(
        r'(?:^|\n)\s*[（(]?[一二三四五六七八九十]{1,3}[）)、.．]\s*\S|'
        r'(?:^|\n)\s*\d{1,3}[.)]\s+\S', re.IGNORECASE)   # ASCII 编号列表(英文材料常见)
    # 解释信号: 定义/机制/因果句式
    _HINT_EXPLAIN = re.compile(
        r'是指|指的是|意为|即[:：]?|原理|机制|原因|之所以|由于|为了.{0,20}(?:实现|达到|保障)|'
        r'如何|怎样|怎么(?:样)?|含义|定义|作用(?:是|在于)|有何[^，。？]{0,8}(?:意义|价值|特点)|影响|'
        r'\bwhat (?:is|are)\b|\brefers? to\b|\b(?:is|are) defined as\b|\bis called\b|\bknown as\b|'
        r'\bwhy\b|\bhow (?:do|does|is|are|can|to)\b|\bmechanism\b|\bprinciple\b|\bcaused by\b|\bdue to\b|\bresults from\b|'
        r'\bdefine\b|\bdescribe\b|\bexplain\b|\bindicate[sd]?\b|\beffect of\b|\bimpact of\b',
        re.IGNORECASE)
    # 建议信号: 祈使/措施词 + 文件型结构词
    _HINT_ADVICE = re.compile(
        r'建议|应当|应(?:该|当)?(?:加强|推进|落实|开展|实施|优化|提升|加快|完善|健全|强化|保障)|'
        r'专项(?:行动|措施)|重点任务|总体目标|总体要求|工作要求|政策措施|行动方案|'
        r'\bshould\b|\bmust\b|\bshall\b|\brecommend(?:s|ed|ation|ations)?\b|\badvise[ds]?\b|'
        r'\baction plan\b|\bmeasures\b|\bguidelines?\b|\bensure that\b|\bbest practices?\b',
        re.IGNORECASE)
    # 比较信号: 对比句式
    _HINT_COMPARE = re.compile(
        r'相比|对比|较之|优于|劣于|胜于|强于|不同于|区别(?:在于)?|差异|权衡|比[^。，,]{1,15}更(?:加)?[^。，,]{0,10}|'
        r'\bcompared (?:with|to)\b|\bin comparison\b|\bversus\b|\bvs\.?(?=\s|$)'
        r'|\bdifferences? (?:between|among)\b|\btrade-?offs?\b|'
        r'\b(?:better|worse|faster|slower|cheaper|safer|more accurate) than\b|\badvantages? (?:of|over)\b',
        re.IGNORECASE)
    # 四类之外: 问的是总结/提取/翻译/改写(没有任何四类意图) —— 不进 COT
    _OUT_OF_SCOPE = re.compile(
        r'总结|摘要|概括|提炼|摘录|逐条列举|列举(?:出)?|列出|提取|摘出|翻译|改写|润色|缩写|简化|梳理|'
        r'转换为|改成|拆成|转成|'
        r'\bsummari[sz]e\b|\bsummary\b|\bextract\b|\blist (?:the|all)\b|\btranslat\w+\b|'
        r'\brewrite\b|\bparaphrase\b', re.IGNORECASE)
    # 事实检索意图(问"哪些/何时/谁/是什么/列出" -> 要的是材料事实, 不属于四类)。
    # 实测: 55 条被判"难以区分"的样本里绝大多数是这类("依据了哪些文件/谁是作者/
    # 何时发布/适合哪些场景"), 它们该归"四类之外"而不是丢给人工审核。
    _RETRIEVE_ASK = re.compile(
        r'哪些|哪几(?:个|项|条|种)|哪(?:四|几|两|三)类|何种|何时|什么时候|哪一年|第几|几位|'
        r'谁(?:是|提出|编写)|何处|哪里|什么是|是什么|是否|有无|多少个|几项|'
        r'什么情况|什么类型|遵循什么|根据关键词|判断(?:该|其|是否)|'
        r'\bwho\b|\bwhen\b|\bwhere\b|\bwhich\b|\blist\b|\bname (?:the|all)\b|'
        r'\bidentify\b|\bclassify\b|\bdetermine\b|\bconvert\b|\bsimplify\b|'
        r'\bwhat (?:is|are|was|were|type|kind|does|questions|architecture|funding)\b|'
        r'\bhow many\b|^\s*(?:is|are|was|were)\b', re.IGNORECASE)
    # 显式类别标记(题面直接点明类别) —— 优先级最高
    _EXPLICIT = (
        ("比较", re.compile(r'对比|比较|差异|优劣|异同|相比|compare|comparison|difference|versus|vs\.?',
                            re.IGNORECASE)),
        ("建议", re.compile(r'建议|推荐|应当|应该|措施|如何改进|怎么改|对策|recommend|advice|should|measures',
                            re.IGNORECASE)),
        ("推理", re.compile(r'计算|求解|算出|求证|证明|推导|推算|算一算|calculate|compute|solve|derive|prove',
                            re.IGNORECASE)),
        ("解释", re.compile(r'解释|说明(?:其)?(?:原因|原理|机制|过程)|为什么|原因(?:是|在于)|原理|机制|'
                            r'explain|why|reason|mechanism|principle', re.IGNORECASE)),
    )
    # 模板套话与来源文件名: 判类前剥离(防来源标签经文件名漏进判类)
    _TEMPLATE_PATS = (
        r'^根据以下关于「[^」]{0,80}」的材料[，,：:]?',
        r'^根据以下材料[，,：:]?', r'^请根据(?:以下)?材料[，,：:]?',
        r'^根据上述材料[，,：:]?', r'^基于以下(?:关于「[^」]{0,80}」的)?材料[，,：:]?',
        r'^围绕材料中的「[^」]{0,80}」[，,：:]?', r'^\(?围绕材料[^）)]{0,80}\)?',
        r'^材料中(?:提到|指出|显示)[，,：:]?',
    )
    _FILENAME_RE = re.compile(r'[^\s，,。；;：:（）()【】\[\]]{0,60}\.(?:txt|md|docx?|pdf|png|jpe?g|html?|pptx?|xlsx?)',
                              re.IGNORECASE)

    def _strip_template(self, instruction: str) -> str:
        """剥掉模板套话与来源文件名, 只留"这条样本问的是什么" —— 判类的对象"""
        s = (instruction or "").strip()
        for _ in range(4):
            before = s
            for pat in self._TEMPLATE_PATS:
                s = re.sub(pat, "", s, count=1).strip()
            if s == before:
                break
        s = self._FILENAME_RE.sub("", s)
        return re.sub(r'\s+', " ", s).strip()

    @staticmethod
    def _hits(instruction: str, output: str, pattern) -> bool:
        return bool(pattern.search(instruction) or pattern.search(output))

    def _judge_sample(self, record: Dict) -> Tuple[Optional[str], str]:
        """判类: 返回 (四类名 | None(四类之外) | UNCERTAIN, 依据)

        判类对象是**这条 SFT 问答对**(题面意图为主, 答案为辅), 不是它所属的文档。
        优先级(按 skill 决策表): 显式类别标记 > 可唯一验证答案(推理) >
        单一题面意图 > 单一答案形态; 信号冲突或全无 -> 难以区分(人工审核)。
        """
        instr = self._strip_template(record.get("instruction") or "")
        output = (record.get("output") or "").strip()
        if not instr:
            return self.UNCERTAIN, "题面为空(模板剥离后无内容)"

        # ① 显式类别标记(命中多个 -> 冲突, 交人工审核, 不按固定优先级硬判)
        explicit = [cls for cls, pat in self._EXPLICIT if pat.search(instr)]
        if len(explicit) == 1:
            return explicit[0], "题面显式类别标记"
        if len(explicit) > 1:
            return self.UNCERTAIN, "题面显式标记冲突: " + "/".join(explicit)

        # ② 存在可唯一验证答案 -> 推理(算式/条件演绎/计数提问 + 可数材料)
        list_items = len(self._HINT_LIST.findall(output or instr))
        if self._HINT_MATH.search(instr) or self._HINT_MATH.search(output):
            return "推理", "存在可重算的算式"
        if self._HINT_REASON.search(instr):
            return "推理", "题面为条件演绎/计算求解"
        if self._COUNT_CUE.search(instr) and (list_items >= 3 or self._COUNT_CUE.search(output)):
            return "推理", "题面要求清点/合计(答案唯一可验证)"

        # ③ 四类之外: 问的是总结/提取/翻译/改写, 或纯事实检索(哪些/何时/谁/是什么),
        #    且题面没有任何四类意图词
        class_intent = bool(explicit) or \
            self._HINT_EXPLAIN.search(instr) or self._HINT_ADVICE.search(instr) or \
            self._HINT_COMPARE.search(instr)
        if self._OUT_OF_SCOPE.search(instr) and not class_intent:
            return None, "四类之外(总结/提取/改写)"
        if self._RETRIEVE_ASK.search(instr) and not class_intent:
            return None, "四类之外(事实检索: 答案只是材料事实的搬运)"

        # ④ 单一题面意图
        ask_hits = [c for c, pat in (("比较", self._HINT_COMPARE), ("建议", self._HINT_ADVICE),
                                    ("解释", self._HINT_EXPLAIN)) if pat.search(instr)]
        if len(ask_hits) == 1:
            return ask_hits[0], "题面意图单一命中: %s" % ask_hits[0]
        if len(ask_hits) > 1:
            return self.UNCERTAIN, "题面意图冲突: " + "/".join(ask_hits)

        # ⑤ 题面无线索 -> 看答案形态(单类命中才用)
        out_hits = [c for c, pat in (("比较", self._HINT_COMPARE), ("建议", self._HINT_ADVICE),
                                    ("解释", self._HINT_EXPLAIN)) if pat.search(output)]
        if len(out_hits) == 1:
            return out_hits[0], "答案形态单一命中: %s" % out_hits[0]
        if len(out_hits) > 1:
            return self.UNCERTAIN, "答案形态冲突: " + "/".join(out_hits)
        return self.UNCERTAIN, "四类信号不足"

    # ================= 【B】回原文搜据 =================
    @staticmethod
    def _first_excerpt(text: str, limit: int) -> str:
        """取材料开头到最近句末的一段(无句末标点时按 limit 截断)。"""
        t = (text or "").strip()
        if len(t) <= limit:
            return t
        head = t[:limit]
        cut = max(head.rfind(c) for c in "。！？!?.")
        return head[:cut + 1] if cut >= 0 else head

    @staticmethod
    def _excerpt_around(text: str, anchor: str, limit: int) -> str:
        """取锚点前半窗 + 锚点及其后内容, 保证锚点在 context 里可见。"""
        t = text or ""
        i = t.find(anchor)
        if i < 0:
            return COTConverter._first_excerpt(t, limit)
        half = max(120, limit // 2)
        return COTConverter._first_excerpt(t[max(0, i - half):], limit)

    def _retrieve_context(self, corpus_text: str, material: str, output: str) -> str:
        """回原文检索: 以样本自带材料为锚点取窗口, 再对答案里的数值做补检索。

        语料原文缺失时退化为样本自带材料 —— 仍受"链只能引用可见内容"的约束。
        """
        from ..utils.validation import numbers_in

        t = (corpus_text or "").strip()
        anchor_src = (material or output or "").strip()
        if not t:
            return anchor_src
        anchor = anchor_src[:40]
        ctx = self._excerpt_around(t, anchor, self.context_window) if anchor \
            else self._first_excerpt(t, self.context_window)
        extra = []
        for num in list(numbers_in(output))[:8]:
            if num in ctx:
                continue
            j = t.find(num)
            if j >= 0:
                extra.append(self._first_excerpt(t[max(0, j - 200):], 400))
        if extra:
            ctx = ctx + "\n……\n" + "\n".join(extra)
        return ctx

    # ================= 【C】生成思维链(以 SFT 为基准 + 校验重试) =================
    _COT_REQ = {
        "推理": "每步写明算式/依据，中间值可验证，最终结论与回答一致；禁止跳步、禁止只给答案。",
        "解释": "因果链清晰，机制有依据，不编造；每步给出理由，结论与回答一致。",
        "建议": "先明确目标与约束，再列出候选方案，给出取舍理由与预期效果，最终收敛为建议结论。",
        "比较": "明确比较维度，逐项事实对比，给出落在具体情境的结论，不能只写“各有优劣”。",
    }
    # skill cot-dataset-generation 第二步的**真实数据集模板** —— 原样嵌入, 不做改写。
    # 每类 = (数据集, 真实样本 JSON 原文, 中文释义[供中文材料参考])
    _COT_SHOT = {
        "推理": ("openai/gsm8k", '{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\\n#### 72"}',
                 "逐句写出算式与中间值(<<算式=结果>> 形式), 最后收敛到最终答案(#### 答案)"),
        "解释": ("allenai/sciq", '{"question": "What type of organism is commonly used in preparation of foods such as cheese and yogurt?", "correct_answer": "mesophilic organisms", "support": "Mesophiles grow best in moderate temperature, typically between 25°C and 40°C (77°F and 104°F). Mesophiles are often found living in or on the bodies of humans or other animals. The optimal growth temperature of many pathogenic mesophiles is 37°C (98°F), the normal human body temperature. Mesophilic organisms have important uses in food preparation, including cheese, yogurt, beer and wine."}',
                 "question/correct_answer/support 三字段: 先给答案, 再用 support 说明机制与依据"),
        "建议": ("jonathanli/legal-advice-reddit", '{"title": "My former employer fired everyone when they had to shutdown so they could avoid paying out sick time... Is this legal?", "body": "The title says most of it. I just heard back from a friend... Cafe Mooby stayed open until the last minute, and then terminated everyone\'s employment... I\'m just curious if it\'s legal, and if the workers can file a wrongful termination class-action lawsuit or something?"}',
                 "真实样本为「情景 + 诉求」(title/body), 链要先列约束再对比方案后给可执行建议"),
        "比较": ("CarperAI/openai_summarize_comparisons", '{"prompt": "SUBREDDIT: r/relationships\\nTITLE: To admit or not to admit snooping...\\nPOST: I [25M] have snooped in the past...", "chosen": "TL;DR: Snooped, found something, should I admit what I found...", "rejected": "TL;DR: I snooped, we talked about it, she wants to work it out, I\'m not sure..."}',
                 "真实样本给出 prompt 与 chosen/rejected 两个候选, 链要按维度逐项对比后下结论"),
    }

    def _build_cot_prompt(self, instruction: str, output: str,
                          context: str, language: str,
                          task_type: Optional[str] = None) -> str:
        """构建 COT 生成 prompt(单模板参数化 + 按类 few-shot, 与 skill 模板同构)"""
        req = "分步骤展开，每个步骤有理有据，最终结论和回答一致。"
        if task_type in self._COT_REQ:
            req = self._COT_REQ[task_type]
        shot = self._COT_SHOT.get(task_type) if task_type else None
        if shot:
            ds, sample, gloss = shot
            if language == "zh":
                shot_line = (f"\n参考真实数据集({ds})的真实样本风格(原样给出, 不要照抄题目):"
                             f"\n{sample}\n(释义: {gloss})")
            else:
                shot_line = (f"\nReference style — a real sample from {ds} "
                             f"(verbatim; do not copy its question):\n{sample}")
        else:
            shot_line = ""
        class_line = (f"\n任务类别：{task_type}\n按该类别的生成要求：\n{req}"
                      if task_type else "")
        basis = ("\n基准(必须遵守)：下面的问答对就是本条的基准 sample —— 只能围绕它写推理过程，\n"
                 "- 不得换题目、不得改答案、不得新增或替换事实与数值；\n"
                 "- instruction/input/output 以基准为准(措辞可微调，核心本义不得变)。")

        if language == "zh":
            return f"""请根据以下问答对，生成详细的推理过程(chain-of-thought)。

原始文本的上下文（供参考）：\n{context[:1200]}

问答对：\n指令：{instruction}\n回答：{output}\n{class_line}\n{shot_line}\n{basis}
纪律（必须遵守）：
- 推理过程的每一步只能引用上方上下文/问答对中**明确出现**的内容；
- 未在上方文本中出现的事实、数字、专名、条目一律禁止引入（禁止凭常识或训练记忆补充）；
- 若信息不足，只基于已给信息推理，禁止外推。
请按以下格式输出（只输出推理过程本身，不要额外解释）：\n1. ...\n2. ...\n3. ...\n因此结论是：{output}"""
        else:
            return f"""Generate a detailed chain-of-thought reasoning process for the following Q&A pair.

Context:
{context[:800]}

Q&A (this is the base sample — reason about it, do not replace it):
Instruction: {instruction}
Answer: {output}
Task category: {task_type or "self-judge"}
Requirements for this category:
{req}
{shot_line}

Rules (mandatory):
- Every step may only cite content that appears above;
- No facts, numbers, names or items outside the text above;
- The instruction/answer stay as given (wording may vary slightly, the meaning must not).
Output format (reasoning process only):
1. ...
2. ...
3. ...
Therefore, the answer is: {output}"""

    def _parse_cot_response(self, response: str):
        """解析COT输出：提取可选的任务类别前缀与推理链文本。
        返回 (task_type or None, 链文本 or None)。"""
        text = (response or "").strip()
        if len(text) < 10:
            return None, None
        m = re.match(r'^类别\s*[:：]?\s*(解释|推理|建议|比较)', text)
        if m:
            return m.group(1), text[m.end():].strip()
        return None, text

    # 列表序号(1. / 一、)不是"链中的数值", 校验前先剥掉 ——
    # 否则解释/建议类的链只因编号带数字就被标成"已验证"
    _LIST_MARKER = re.compile(r'^\s*(?:\d{1,3}[.)、]|[一二三四五六七八九十]{1,3}[、.)])\s*', re.M)

    @classmethod
    def _strip_list_markers(cls, chain: str) -> str:
        return cls._LIST_MARKER.sub("", chain or "")

    def _generate_chain(self, instruction: str, output: str, material: str,
                        context: str, language: str, task_type: str):
        """生成并校验思维链。返回 (链 | None, 答案已验证, 链已验证)

        "已验证"的口径: 链里**确实有可核数值**(剥离列表序号后)且全部能在材料/回答中
        找到出处 -> reasoning_correct=true; 没有可核数值 -> null(未验证, 不猜)。

        校验不过按 llm_max_attempts 重试; 重试后仍不过 -> 返回 None(调用方丢弃)。
        """
        from ..utils.validation import math_chain_check, numeric_consistency, numbers_in

        attempts = max(1, self.llm_max_attempts)
        for attempt in range(attempts):
            try:
                resp = self.llm_generator.generate(
                    self._build_cot_prompt(instruction, output, context, language, task_type),
                    system_prompt="你是训练数据标注员：只依据给定材料写推理过程，绝不引入材料外的事实。")
            except Exception as e:                      # 网络/限流等瞬时错误 -> 重试
                logger.warning("COT 生成调用异常(第 %d/%d 次, %s): %s",
                               attempt + 1, attempts, task_type, e)
                continue
            _, chain = self._parse_cot_response(resp)
            if not chain or len(chain.strip()) < 40:
                continue                                # 过短/空 -> 重试
            bare = self._strip_list_markers(chain)
            nums = numbers_in(bare)
            if nums and not numeric_consistency(output, context, bare, material):
                continue                                # 链中数值在材料里查不到出处
            if task_type == "推理":
                if math_chain_check(chain) is False:
                    continue                            # 算式重算失败 -> 重试, 仍错则丢弃
                return chain, True, True
            return chain, False, bool(nums)
        return None, False, False

    def _build_sample(self, record: Dict, task_type: str, source: str,
                      corpus_text: str, corpus_type: str) -> Optional[Dict]:
        """一条 SFT 样本 -> 一条 COT 样本(素材不得自造: instruction/input/output 原样沿用)"""
        instruction = (record.get("instruction") or "").strip()
        material = (record.get("input") or "").strip()
        output = (record.get("output") or "").strip()
        if not instruction or not output:
            return None
        language = detect_language(instruction + "\n" + output)
        context = self._retrieve_context(corpus_text, material, output)
        chain, answer_ok, reasoning_ok = self._generate_chain(
            instruction, output, material, context, language, task_type)
        if not chain:
            self.stats["dropped"] += 1
            self.stats["dropped_reasons"]["生成/校验不过"] = \
                self.stats["dropped_reasons"].get("生成/校验不过", 0) + 1
            return None
        return {
            "instruction": instruction,
            "input": material,
            "output": output,
            "chain_of_thought": chain,
            "task_type": task_type,
            # verifiable = 该条是否存在可唯一验证的答案
            "verifiable": task_type == "推理",
            # 两字段**只在机器验证通过时填 true**, 未验证为 null(《数据集介绍.md》4.2/4.3)
            "final_answer_correct": True if answer_ok else None,
            "reasoning_correct": True if reasoning_ok else None,
            # 溯源: 派生自哪条 SFT 样本 + 哪个语料
            "source": {
                "sft_id": record.get("id"),
                "corpus_file": os.path.basename(source or ""),
                "corpus_type": corpus_type or "",
            },
        }

    # ================= 对外入口 =================
    def convert_sft_batch(self, source: str, records: List[Dict],
                          corpus_text: str = "", corpus_type: str = "") -> List[Dict]:
        """把一份语料的 SFT 样本转成 COT 样本(判类 -> 搜据 -> 生成 -> 校验)。

        **整份来源为单位**调用: 与流水线的来源级并发/写盘集中化一致。
        """
        samples: List[Dict] = []
        if not records:
            return samples
        if not self.llm_generator:
            logger.warning("[COT] 未启用 LLM 生成器, 跳过来源 %s", source)
            return samples

        self.stats["sft_total"] += len(records)
        review = []
        for record in records:
            task_type, why = self._judge_sample(record)
            if task_type is None:
                self.stats["out_of_scope"] += 1
                logger.debug("[COT] 四类之外: %s | %s", why, (record.get("instruction") or "")[:40])
                continue
            if task_type == self.UNCERTAIN:
                self.stats["uncertain"] += 1
                review.append((record, "判类难以区分(%s)" % why))
                continue
            self.stats["candidates"] += 1
            sample = self._build_sample(record, task_type, source, corpus_text, corpus_type)
            if sample:
                samples.append(sample)
                self.stats["by_class"][task_type] = self.stats["by_class"].get(task_type, 0) + 1

        if review:
            logger.warning("[COT] %s: 判类难以区分 %d 条 -> 人工审核", source, len(review))
            self._flush_review(review)

        if self.max_samples > 0:
            samples = samples[:self.max_samples]

        # 确定性硬校验(数学题重算 / 隐私红线 / 代码运行): 不合格直接剔除
        samples = self._hard_validate(samples)

        # LLM 语义抽检(幻觉/跳步/类别匹配; 争议样本移出主集并导出人工审核)
        if self.sample_check_ratio > 0 and samples:
            samples = self._semantic_sample_check(samples)

        self.stats["generated"] += len(samples)
        return samples

    # ================= 校验 / 统计 / 审核导出 =================
    def _hard_validate(self, samples: List[Dict]) -> List[Dict]:
        """确定性硬校验（《数据集介绍.md》6.3）：数学题重算、隐私红线、代码运行。
        不合格样本直接剔除（机器可复现的客观判定，无需人工）。"""
        from ..utils.validation import math_chain_check, privacy_scan, code_exec_check
        bad_reasons = {}

        for i, s in enumerate(samples):
            # 1. 数学题重算：推理类且链中含显式等式，验算错误即剔除
            if s.get("task_type") == "推理":
                r = math_chain_check(s.get("chain_of_thought", ""))
                if r is False:
                    bad_reasons[i] = "算式重算失败"
                    continue
            # 2. 隐私红线：input/output 命中手机号/身份证/银行卡 → 直接剔除
            hits = privacy_scan((s.get("input") or "") + " " + (s.get("output") or ""))
            if hits:
                bad_reasons[i] = f"隐私未脱敏({','.join(sorted(hits))})"
                continue
            # 3. 代码题运行（默认关）：含代码块且被抽中安全门槛 → 输出不符剔除
            if self.verify_code_exec:
                r = code_exec_check((s.get("input") or "") + " " + (s.get("chain_of_thought") or ""),
                                    s.get("output") or "")
                if r is False:
                    bad_reasons[i] = "代码运行失败或输出不符"
                    continue

        if bad_reasons:
            counts = {}
            for r in bad_reasons.values():
                counts[r] = counts.get(r, 0) + 1
            logger.warning("COT 确定性校验剔除 %d 条: %s", len(bad_reasons), counts)
            self.stats["dropped"] += len(bad_reasons)
            for r, c in counts.items():
                self.stats["dropped_reasons"][r] = self.stats["dropped_reasons"].get(r, 0) + c
            samples = [s for i, s in enumerate(samples) if i not in bad_reasons]
        return samples

    @staticmethod
    def _consistency_check(output: str, material: str, chain: str,
                           full_material: str = '') -> bool:
        """数字一致性硬查：output 中的数值必须能在材料/推理链中找到出处。
        材料外数值是幻觉/答非所问的强证据；output 无数值则放行交语义抽检。
        full_material=全文, 节选内未检出时兜底复查(防节选截断假阳)。"""
        from ..utils.validation import numeric_consistency
        return numeric_consistency(output, material, chain, full_material)

    def _distribution(self, samples: List[Dict]) -> Dict:
        dist = {}
        for s in samples:
            dist[s.get("task_type", "?")] = dist.get(s.get("task_type", "?"), 0) + 1
        return dist

    def report_distribution(self, samples: List[Dict]) -> Dict:
        """四类分布**只统计只报告**: 不裁剪、不改判、不删样本(实事求是)。

        差额归因到素材层(哪些 SFT 样本本身不属于该类), 不在生成侧凑数。
        """
        dist = self._distribution(samples)
        total = len(samples)
        n = total or 1
        parts = []
        for cls in self.CLASSES:
            got = dist.get(cls, 0) / n
            target = self._DIST_TARGET[cls]
            if total == 0:
                flag = "无样本"
            elif abs(got - target) <= self.distribution_tolerance:
                flag = "达标"
            else:
                flag = "偏低" if got < target else "偏高"
            parts.append("%s %d(%.1f%%/目标%.0f%% %s)" % (cls, dist.get(cls, 0), got * 100,
                                                         target * 100, flag))
        logger.info("COT 四类分布(%d 条, 只统计不裁剪): %s", total, " | ".join(parts))
        logger.info("COT 判类统计: SFT %d 条 -> 候选 %d, 四类之外 %d, 难以区分 %d(转人工), "
                    "生成 %d, 丢弃 %d %s",
                    self.stats["sft_total"], self.stats["candidates"], self.stats["out_of_scope"],
                    self.stats["uncertain"], self.stats["generated"], self.stats["dropped"],
                    self.stats["dropped_reasons"] or "")
        return dist

    # ============ 样本级 LLM 语义抽检(幻觉/跳步/类别匹配/有害推理) ============
    _CHECK_PROMPT = """请从严审查以下思维链训练样本。四项检查：
1. output 是否严格基于 input 材料(无材料外事实/数字,无幻觉)
2. chain_of_thought 是否与 output 一致、无跳步、无编造
3. task_type 类别与实际推理类型是否匹配
4. 无有害/违法/危险推理内容

注意：input 允许是材料节选——只要 input 内条目完整、且 output/链只引用
input 中可见的内容，节选本身不构成不合格项（不要因"材料不完整"判 fail）。
判 fail 必须给出具体证据（哪一步与材料不符/哪个数字无出处/哪处跳步），无证据一律 pass。
说明: 基于材料逐条核对并计数(如"清点共几个条目")属于可接受的推理类形态
(task_type=推理), 计数与材料的编号条目一一对应即合格, 不要以"仅是计数/信息提取"判 fail。
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
                        "chain_of_thought": s.get("chain_of_thought") or "",
                        "task_type": s.get("task_type", ""),
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
        """争议样本落盘成待人工审核清单（review_output 配置时）；空路径则仅记日志。
        列出审核字段：样本全文 + 触发原因 + 时间戳；人工双人审核后决定取舍/修正。"""
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