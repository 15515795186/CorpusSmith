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

    def __init__(self, max_samples: int = 0, llm_generator=None,
                 sample_check_ratio: float = 0.0,
                 distribution_enforce: bool = True,
                 distribution_tolerance: float = 0.05,
                 review_output: str = "",
                 verify_code_exec: bool = False,
                 llm_max_attempts: int = 1):
        self.max_samples = max_samples
        self.llm_generator = llm_generator
        # 样本级校验（转换后，从严）
        self.sample_check_ratio = sample_check_ratio   # LLM 语义抽检比例（0=关）
        self.distribution_enforce = distribution_enforce  # 四类分布硬门禁
        self.distribution_tolerance = distribution_tolerance
        self.review_output = review_output         # 争议样本导出路径（人工双人审核）
        self.verify_code_exec = verify_code_exec   # 代码题运行验证（默认关）
        self.llm_max_attempts = llm_max_attempts   # 拒绝采样: 仅推理类硬校验失败时重试上限
        # 目标四类分布（《数据集介绍.md》4.5）与门禁阈值
        self._DIST_TARGET = {"解释": 0.20, "推理": 0.30, "建议": 0.30, "比较": 0.20}

    def convert(self, result: ReaderResult) -> List[Dict]:
        """将读取结果转换为COT样本列表（遵守 skill 四类流程：预判→按类模板/LLM→分布统计）"""
        if not result.success or not result.text:
            return []

        text = result.text.strip()
        # 剥离扫描/转换残留的 markdown 加粗标记（保留内容，去除 ** 对）
        text = re.sub(r'\*\*([^*\n]+?)\*\*', r'\1', text)
        samples = []

        # 四类预判（规则信号；None=信号弱，由 LLM 通道兜底判断）
        task_type = self._classify_text(text)

        # 1. 结构化内容模板（推理·清点 / 解释·流程 / 因果 / 三步）
        step_samples = self._extract_step_cot(text, result)
        samples.extend(step_samples)
        definition_samples = self._extract_definition_cot(text, result)
        samples.extend(definition_samples)

        # 2. 建议类：模板通道停用——措施/要求开列条目的"建议"样本本质是复制清单，
        #    审查员反复判"仅信息提取,与标注类型不匹配"；真正给出建议的样本由
        #    LLM 通道(_generate_cot_with_llm, 对应 task_type=建议)产出。
        #    advice_samples = self._extract_advice_cot(text, result)
        #    samples.extend(advice_samples)

        # 3. 比较类模板（同句对比结构 → 比较链）
        compare_samples = self._extract_compare_cot(text, result)
        samples.extend(compare_samples)

        # 4. LLM 通道: 按四类目标配比补齐样本。
        #    《数据集介绍.md》4.5 要求整体分布 20/30/20/30(±5%), 而四类预判是高优先级
        #    单类别判定: 语料类别集中时(实测 79% 材料判推理、比较类恒为 0)无论重跑多少次
        #    都到不了目标。故对每份材料按"体量 ↔ 条数"出样本、按 20/30/20/30 分配四类,
        #    每类用自己的真实样本 few-shot 与该类的抽取式问法。
        #    条数**不固定**: 长材料出更多、短材料出更少(见 _source_budget)。
        if self.llm_generator:
            llm_samples = self._generate_cot_with_llm(text, result, task_type, samples)
            samples.extend(llm_samples)

        if self.max_samples > 0:
            samples = samples[:self.max_samples]

        # 确定性硬校验（数学题重算 / 隐私红线 / 代码运行）: 不合格直接剔除
        samples = self._hard_validate(samples)

        # 分布硬门禁: 超配裁剪 + 欠配告警(样本 <10 时不做裁剪, 仅统计)
        if self.distribution_enforce and len(samples) >= 10:
            samples = self._enforce_distribution(samples)
        else:
            dist = self._distribution(samples)
            logger.info("COT 四类分布: %s | 目标: 解释20%%/推理30%%/建议30%%/比较20%%", dist)

        # LLM 语义抽检(幻觉/跳步/类别匹配; 争议样本导出人工审核)
        if self.sample_check_ratio > 0 and self.llm_generator and samples:
            samples = self._semantic_sample_check(samples)

        return samples

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
            samples = [s for i, s in enumerate(samples) if i not in bad_reasons]
        return samples

    def _distribution(self, samples: List[Dict]) -> Dict:
        dist = {}
        for s in samples:
            dist[s.get("task_type", "?")] = dist.get(s.get("task_type", "?"), 0) + 1
        return dist

    def _enforce_distribution(self, samples: List[Dict]) -> List[Dict]:
        """分布硬门禁：超配类裁剪到上限(按产出顺序保留)、欠配类告警。"""
        total = len(samples)
        # 上限 = 目标占比×总数 进一 +1(容忍 round 抖动), 即不超出 ±5% 的配额
        # 上限 = (目标占比 + 容忍) × 总数, 进一取整 —— 只裁**超出容忍**的超配,
        # 不把高价值样本无端裁掉(旧口径"目标占比+1"对长材料偏紧)
        caps = {t: max(1, int((target + self.distribution_tolerance) * total + 0.999))
                for t, target in self._DIST_TARGET.items()}
        dist = self._distribution(samples)
        overflow = {t: n - caps[t] for t, n in dist.items()
                    if t in caps and n > caps[t]}
        if overflow:
            keep, kept = {}, []
            for s in samples:
                tt = s.get("task_type", "?")
                if tt in overflow and keep.get(tt, 0) >= caps[tt]:
                    continue
                keep[tt] = keep.get(tt, 0) + 1
                kept.append(s)
            samples = kept
            logger.warning("COT 分布硬门禁: 超配裁剪 %s, 裁剪后 %s",
                           overflow, self._distribution(samples))

        dist = self._distribution(samples)
        total = len(samples) or 1
        for t, target in self._DIST_TARGET.items():
            ratio = dist.get(t, 0) / total
            if ratio < target - self.distribution_tolerance:
                logger.warning("COT 分布欠配: %s 类占比 %.0f%% (目标 %.0f%%), 建议补充该类样本",
                               t, ratio * 100, target * 100)
        logger.info("COT 四类分布(硬门禁后): %s | 目标: 解释20%%/推理30%%/建议30%%/比较20%%", dist)
        return samples

    _NUM_RE = None  # 数值提取已收敛到 utils/validation.py

    @staticmethod
    def _numbers_in(text: str):
        """提取文本中的数值 token(全角数字归一为半角)。"""
        from ..utils.validation import numbers_in
        return numbers_in(text)

    @staticmethod
    def _consistency_check(output: str, material: str, chain: str,
                           full_material: str = '') -> bool:
        """数字一致性硬查：output 中的数值必须能在材料/推理链中找到出处。
        材料外数值是幻觉/答非所问的强证据；output 无数值则放行交语义抽检。
        full_material=全文, 节选内未检出时兜底复查(防节选截断假阳)。"""
        from ..utils.validation import numeric_consistency
        return numeric_consistency(output, material, chain, full_material)

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

    # ============ 四类预判（skill: cot-dataset-generation 决策表落地）============
    # 推理信号：显式算式 / 合计计数 / ≥3 条编号列表（清点可数）。目标数字（如
    # “下降13.5%”）不算推理信号——它是建议/政策文本的常客，误判会把政策文本
    # 整体归入推理类。
    # 信号按 skill 决策表实现，**不区分语种**：同一套语义特征，中英各有对应形态
    # （英文形态此前缺失 → 英文材料整类判空，见 commit 说明）。优先序不变：
    # 推理 > 解释 > 建议 > 比较。
    # 推理·算式信号: **只认真算式** —— 带等号 / 乘除 / 倍数。
    # 旧模式 "数字[+-*/]数字" 会把日期("2024-03-19")与数值区间("40-50%")当算术,
    # 实测 80 份语料中 17 份(21%)因此被错判为推理, 是四类分布失衡的首要原因。
    _HINT_MATH = re.compile(
        r'\d+\s*[+\-×*/÷]\s*\d+\s*[=＝]|'
        r'\d+\s*[×*/÷]\s*\d+|'
        r'\d+\s*[加减乘除]\s*\d+\s*[等为得]|'
        r'[一二三四五六七八九十百千]+倍', re.IGNORECASE)
    # 计数语义(决策表的"清点/计数条目"): 编号列表**必须配它**才算推理,
    # 否则"只列不数"的政策清单会被一律判成推理(实测 58% 材料有 ≥3 编号条目,
    # 但同时带计数语义的只有 15%)。
    _COUNT_CUE = re.compile(
        r'[共合计]计?\s*[一二三四五六七八九十百\d]+\s*[个项条种部章节]|'
        r'几(?:个|项|条|种|类|部分)|多少个|数量统计|条目数|'
        r'\bhow many\b|\b(?:total|sum|altogether)\b', re.IGNORECASE)
    # 推理·逻辑演绎信号: 只认条件式演绎; 单"因此/therefore"是结论词不算推理依据
    _HINT_REASON = re.compile(
        r'如果[^。\n]{0,20}(?:那么|就|则)|若[^。\n]{0,20}(?:则|就)|假设[^。\n]{0,20}(?:则|就)|'
        r'\bif\b[^.\n]{0,60}\bthen\b|'
        r'由此(?:可见|可知)|推导(?:得出|可得)', re.IGNORECASE)
    _HINT_LIST = re.compile(
        r'(?:^|\n)\s*[（(]?[一二三四五六七八九十]{1,3}[）)、.．]\s*\S|'
        r'(?:^|\n)\s*\d{1,3}[.)]\s+\S', re.IGNORECASE)   # ASCII 编号列表（英文材料常见）
    # 解释信号：定义/机制/因果句式
    _HINT_EXPLAIN = re.compile(
        r'是指|指的是|意为|即[:：]?|原理|机制|原因|之所以|由于|为了.{0,20}(?:实现|达到|保障)|'
        r'\bwhat (?:is|are)\b|\brefers? to\b|\b(?:is|are) defined as\b|\bis called\b|\bknown as\b|'
        r'\bwhy\b|\bhow does\b|\bmechanism\b|\bprinciple\b|\bcaused by\b|\bdue to\b|\bresults from\b',
        re.IGNORECASE)
    # 建议信号：祈使/措施词 + 文件型结构词
    _HINT_ADVICE = re.compile(
        r'建议|应当|应(?:该|当)?(?:加强|推进|落实|开展|实施|优化|提升|加快|完善|健全|强化|保障)|'
        r'专项行动|重点任务|总体目标|总体要求|工作要求|政策措施|行动方案|'
        r'\bshould\b|\bmust\b|\bshall\b|\brecommend(?:s|ed|ation|ations)?\b|\badvise[ds]?\b|'
        r'\baction plan\b|\bmeasures\b|\bguidelines?\b|\bensure that\b|\bbest practices?\b',
        re.IGNORECASE)
    # 比较信号：对比句式
    _HINT_COMPARE = re.compile(
        r'相比|对比|较之|优于|劣于|胜于|强于|不同于|区别(?:在于)?|差异|权衡|比[^。，,]{1,15}更(?:加)?[^。，,]{0,10}|'
        r'\bcompared (?:with|to)\b|\bin comparison\b|\bversus\b|\bvs\.?(?=\s|$)|'
        r'\bdifferences? (?:between|among)\b|\btrade-?offs?\b|'
        r'\b(?:better|worse|faster|slower|cheaper|safer|more accurate) than\b|\badvantages? (?:of|over)\b',
        re.IGNORECASE)

    def _classify_text(self, text: str) -> Optional[str]:
        """规则预判文本所属 COT 类别（优先级：推理 > 解释 > 建议 > 比较）。
        返回 None 表示信号弱，由 LLM 通道兜底判断。"""
        if not text:
            return None
        # 推理 = 真算式 / 条件演绎 / (≥3 编号条目 **且** 有计数语义)。
        # "只列不数"的编号清单不再判推理, 让位给 解释/建议/比较, 避免单一类别吃掉全库。
        if (self._HINT_MATH.search(text)
                or self._HINT_REASON.search(text)
                or (len(self._HINT_LIST.findall(text)) >= 3
                    and self._COUNT_CUE.search(text))):
            return "推理"
        if self._HINT_EXPLAIN.search(text):
            return "解释"
        if self._HINT_ADVICE.search(text):
            return "建议"
        if self._HINT_COMPARE.search(text):
            return "比较"
        return None

    # 条目尾部被正则吞入的下级编号标题（如 "…全覆盖。 （三）钢铁行业节能降碳行动"）
    _TRAIL_HEADING = re.compile(r'\s*[（(][一二三四五六七八九十]{1,3}[）)]\s*[^。\n]{1,40}$')

    @staticmethod
    def _clean_desc_tail(desc: str) -> str:
        """剥掉条目尾部被吞入的下级标题残行；仅当裁剪后剩完整句才生效（保守）。"""
        d = desc.strip()
        m = COTConverter._TRAIL_HEADING.search(d)
        if m:
            before = d[:m.start()].rstrip(' ，,、；;：:')
            if before and before[-1] in '。！？!?。':
                return before
        return d

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
                # kept 行号用连续序号（k+1）：跨小节合并的编号（1..3 1..3）重排为
                # 1..6，杜绝"重复编号"语义歧义，链与计数一一对应
                kept, used = [], 0
                for num, desc in steps:
                    desc_clean = COTConverter._clean_desc_tail(re.sub(r'\s+', ' ', desc.strip()))
                    if not desc_clean:
                        continue
                    line = f"{len(kept) + 1}. {desc_clean}"
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
                        "task_type": "解释",
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
                    "task_type": "推理",
                })

                if len(samples) >= 10:
                    return samples

        # 中文顿号/括号编号（一、… / （一）…）的清点通道。
        # 只产"条目清点"不产"流程逐步"：中文编号多为并行宣告（白皮书小节体），
        # 无时序证据时对它讲"完整流程"就是语义错配；但"数数"对任何清单成立。
        # 守卫回填自 instruction 列举通道（实测 578 条零误报的那套），并严一档
        # （中位长≥12）：中文编号更易被用作章节短标题（"一、历史背景"）。
        if len(samples) < 10:
            samples.extend(self._cn_numbered_inventory(text, source))
        return samples

    @staticmethod
    def _cn_value(s: str):
        """中文序数词 → int（一..九、十、十一..十九、二十、二十一..九十九）"""
        d = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
             '六': 6, '七': 7, '八': 8, '九': 9}
        if s in d:
            return d[s]
        if s == '十':
            return 10
        m = re.fullmatch(r'十([一二三四五六七八九])', s)
        if m:
            return 10 + d[m.group(1)]
        m = re.fullmatch(r'([一二九])十([一二三四五六七八九])?', s)
        if m:
            return d[m.group(1)] * 10 + (d[m.group(2)] if m.group(2) else 0)
        return None

    @staticmethod
    def _cn_clean_run(cand):
        """中文编号序列校验：+1 递增切段取最长段；条数 2~12；描述中位长 ≥12"""
        if not cand:
            return None
        runs, cur = [], [cand[0]]
        for prev, nxt in zip(cand, cand[1:]):
            pv = COTConverter._cn_value(prev[0])
            nv = COTConverter._cn_value(nxt[0])
            if pv is not None and nv == pv + 1:
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
        if lens[len(lens) // 2] < 12:
            return None
        return steps

    def _cn_numbered_inventory(self, text: str, source: str) -> List[Dict]:
        samples = []
        pats = [
            r'(?:^|\n)\s*([一二三四五六七八九十]{1,3})、\s*(.+?)(?=\n\s*[一二三四五六七八九十]{1,3}、|\n\n|\Z)',
            r'(?:^|\n)\s*[（(]([一二三四五六七八九十]{1,3})[）)]\s*(.+?)(?=\n\s*[（(][一二三四五六七八九十]{1,3}[）)]|\n\n|\Z)',
        ]
        for pat in pats:
            cand = re.findall(pat, text, re.DOTALL)
            steps = self._cn_clean_run(cand)
            if not steps:
                continue
            # 标题组排除：组内任一条目含内嵌数字子列表（如"二、…行动 1.…2.…3.…"，
            # 标题行 + 数字条目混排）→ 交给数字编号通道清点，避免把章节标题计入；
            # 无句号且 ≤25 字的短行视为标题行，剔除出计数。
            joined = "\n".join(d for _, d in steps)
            if re.search(r'(?:^|\n)\s*\d+[.、．]', joined):
                continue
            steps = [st for st in steps
                     if not (('。' not in st[1]) and len(st[1].strip()) <= 25)]
            if len(steps) < 2:
                continue
            kept, used = [], 0
            for num, desc in steps:
                desc_clean = COTConverter._clean_desc_tail(re.sub(r'\s+', ' ', desc.strip()))
                if not desc_clean:
                    continue
                line = f"{num}、{desc_clean}"
                if used + len(line) + 1 > 1500:
                    break
                kept.append(line)
                used += len(line) + 1
            if len(kept) < 2:
                continue
            topic = self._find_step_context(text, (steps[0][1] or "")[:20])
            if len(topic) > 50:
                topic = topic[:50] + "..."
            topic = f"「{topic}」" if topic else ""
            cot_steps = "\n".join(kept)
            samples.append({
                "instruction": f"根据以下材料，清点{topic}总共列出了几个条目？请逐条核对。" if topic else
                               "根据以下材料，清点其中总共列出了几个条目？请逐条核对。",
                "input": cot_steps,
                "chain_of_thought": f"逐条清点材料：\n{cot_steps}\n核对编号后共{len(kept)}个条目。",
                "output": f"{topic}共列出{len(kept)}个条目。" if topic else f"材料共列出{len(kept)}个条目。",
                "source": source,
                "task_type": "推理",
            })
            if len(samples) >= 2:  # 顿号/括号两种风格各最多一条，防单文档挤占
                break
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
            # 多段完整递增编号（1..3 标题 1..3 = 两个真实小节 6 条）合并计数；
            # 单条碎片段（截断/目录 1,2,3,1,2,3）不合并，取最长段防目录噪声
            merged = [runs[0]]
            for r in runs[1:]:
                if len(r) >= 2 and len(merged[-1]) >= 2:
                    merged[-1] = merged[-1] + r
                else:
                    merged.append(r)
            steps = max(merged, key=len)
        # 步骤数守卫：≤40（多小节编号合并后 20-30 条是常态；目录/章节名噪音
        # 由"中位长≥10"守卫拦截，预算截断兜底 kept ≤ ~10 条，材料与计数必然一致）
        if not (2 <= len(steps) <= 40):
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

        # 因果/条件句式提取。同句约束保留（开卷契约：推理两端与连接词必须在同一
        # 句内——跨句时中间论证链断裂，题面无法自包含），但关联词对从最初仅 3 组
        # 扩到常见书面组：由于…因此、只有…才 等此前都是被白白丢弃的真实推理结构。
        patterns = [
            (r'因为(.+?)，所以(.+?)[。.]', "因果关系推理"),
            (r'因为(.+?)，因此(.+?)[。.]', "因果关系推理"),
            (r'由于(.+?)，(?:所以|因此)(.+?)[。.]', "因果关系推理"),
            (r'既然(.+?)，(?:那么|就|则)(.+?)[。.]', "因果推理"),
            (r'如果(.+?)，(?:那么|就|则)(.+?)[。.]', "条件推理"),
            (r'只要(.+?)，(?:那么|就|则)(.+?)[。.]', "条件推理"),
            (r'只有(.+?)，(?:才能|才)(.+?)[。.]', "条件推理"),
            (r'首先[,，](.+?)[,，]其次[,，](.+?)[,，]最后[,，](.+?)[。.]', "三步推理"),
        ]
        # 自包含守卫：前提/结论不得含句末标点或分号（防贪婪配对吞并下一环推理链）、
        # 不得含引号（防把转述句嵌套切出来）、长度上下限；前提不得以指代词或
        # 第一/二人称引导句开头（"这/该…"悬空指代、"我们看一看…"是叙事邀请不是事实前提）
        _bad = re.compile(r'[。！？!?；;“”‘’「」"\']')
        _anaphora = re.compile(r'^(这|那|该|此|其|他|她|它|并|也|而且|但|而|所以|因此|可见|总之'
                               r'|我们|我|你|您|大家|请|试|请问)')
        seen = set()

        for pattern, logic_type in patterns:
            for m in re.finditer(pattern, text):
                groups = m.groups()
                if m.group(0) in seen:
                    continue
                if len(groups) == 2:
                    # 配对可能把句首从句逗号留在组边缘，先剥再校验
                    premise, conclusion = (re.sub(r'^[，,、；;：:\s]+', '', g.strip()) for g in groups)
                    if not (8 <= len(premise) <= 120 and 8 <= len(conclusion) <= 150):
                        continue
                    if _bad.search(premise) or _bad.search(conclusion) or _anaphora.match(premise):
                        continue
                    # 因果/条件类：已知条件已写进题面 → 题目自包含，input 空合规
                    seen.add(m.group(0))
                    samples.append({
                        "instruction": f"根据已知信息，分析{premise}会得出什么结论？请给出推理过程。",
                        "input": "",
                        "chain_of_thought": f"推理过程：\n1. 已知条件：{premise}\n"
                                              f"2. 逻辑分析：根据{logic_type}原则\n"
                                              f"3. 推理结论：{conclusion}",
                        "output": conclusion,
                        "source": source,
                        "task_type": "推理",
                    })
                elif len(groups) == 3:
                    first, second, third = (g.strip() for g in groups)
                    if any(_bad.search(g) for g in (first, second, third)):
                        continue
                    seen.add(m.group(0))
                    # 旧题面"请逐步分析完整过程"没有对象；现在原句进 input，题面自包含
                    samples.append({
                        "instruction": "根据以下材料，按顺序逐步分析其完整过程。",
                        "input": m.group(0)[:600].strip(),
                        "chain_of_thought": f"第一步骤：{first}\n"
                                              f"第二步骤：{second}\n"
                                              f"第三步骤：{third}",
                        "output": third,
                        "source": source,
                        "task_type": "解释",
                    })

                if len(samples) >= 10:
                    return samples

        return samples

    # 建议类：动作词开列的条目（推动/加强/推进…），中位长守卫复用列举通道口径
    _ADVICE_ITEM = re.compile(
        r'(?:^|\n)\s*(?:\d+[.、．]|[（(]?[一二三四五六七八九十]{1,3}[）)]?[、.．])'
        r'\s*((?:推动|加强|推进|落实|开展|实施|优化|提升|加快|完善|健全|强化|支持|鼓励|'
        r'引导|保障|严格|建立|深化|统筹|协调)[^。\n]{6,80})')

    def _extract_advice_cot(self, text: str, result: ReaderResult) -> List[Dict]:
        """建议类模板：措施/要求开列条目 → 建议链（目标约束→候选→取舍→结论）。"""
        samples = []
        source = os.path.basename(result.filepath)
        seen, kept = set(), []
        for m in self._ADVICE_ITEM.finditer(text):
            it = m.group(1).strip()
            if it not in seen:
                seen.add(it)
                kept.append(it)
        if len(kept) < 3:
            return []
        lens = sorted(len(d) for d in kept)
        if lens[len(lens) // 2] < 12:
            return []  # 短词占多数：目录/章节标题不是实质措施

        budget, lines = 1500, []
        for i, d in enumerate(kept):
            line = f"{i + 1}. {d}"
            if sum(len(l) for l in lines) + len(line) > budget:
                break
            lines.append(line)
        if len(lines) < 3:
            return []

        context = self._find_step_context(text, kept[0][:20])
        topic = f"「{context}」" if context and len(context) <= 50 else ""
        conclusion = "；".join(lines) + "。"  # 完整列举预算内条目, 不以 "等。" 敷衍
        # 链引用具体条目与材料语境(模板本无法真正推理, 但忠实于材料)
        listed = "\n".join(f"   - {l}" for l in lines[:5]) + ("\n   ..." if len(lines) > 5 else "")
        samples.append({
            "instruction": (f"根据以下材料，列出{topic}提出的具体可操作措施。"
                            if topic else "根据以下材料，列出其中提出的具体可操作措施。"),
            "input": "\n".join(lines),
            "chain_of_thought": (f"分析思路：\n"
                                 f"1. 明确范围：依据材料{topic or '总体要求'}列举措施清单\n"
                                 f"2. 候选措施（共{len(lines)}条）:\n{listed}\n"
                                 f"3. 取舍：按材料所列优先次序直接采纳\n"
                                 f"因此措施清单为：{conclusion}"),
            "output": conclusion,
            "source": source,
            # 模板只能列举措施(信息提取), 不冒充"建议"类——真正给出建议的样本
            # 由 LLM 通道产出, task_type 与指令语义保持一致
            "task_type": "推理",
        })
        return samples

    # 比较类：显式对比句「A(与|和)B 相比/对比，更/较为…」。
    # 守卫：必须出现显式比较标记(相比/对比)——"总量和强度调控,重点控制…"这类
    # 并列宾语没有标记，不会再被"更高水平"之类的程度词意外命中；A/B 不含顿号
    # 逗号(并列列举特征)；长度上限防吞整句。
    _COMPARE_SENT = re.compile(
        r'([^。，,\n]{4,28}?)(?:与|和)([^。，,\n]{2,28}?)(?:相比|对比)[，,]'
        r'([^。\n]{1,60}(?:更|较为|相对|反而|更需|更应)[^。\n]{0,40})[。.]')

    def _extract_compare_cot(self, text: str, result: ReaderResult) -> List[Dict]:
        """比较类模板：同句对比 → 比较链（维度→事实对比→结论）。"""
        samples = []
        source = os.path.basename(result.filepath)
        for m in self._COMPARE_SENT.finditer(text):
            a, b, rest = (g.strip() for g in m.groups())
            if not (2 <= len(a) <= 40 and 2 <= len(b) <= 40 and len(rest) >= 6):
                continue
            samples.append({
                "instruction": f"比较“{a}”与“{b}”的差异，说明各自的特点与适用情境。",
                "input": m.group(0)[:600],
                "chain_of_thought": (f"1. 比较维度：按材料中的对比点展开\n"
                                     f"2. 事实对比：{a} 与 {b} 的差异——{rest}\n"
                                     f"3. 结论：各自的适用情境不同，结论落在具体语境"),
                "output": rest,
                "source": source,
                "task_type": "比较",
            })
            if len(samples) >= 2:  # 防单文档同句重复挤占
                break
        return samples

    def _generate_cot_with_llm(self, text: str, result: ReaderResult,
                               task_type: Optional[str] = None,
                               existing: Optional[List[Dict]] = None) -> List[Dict]:
        """用LLM生成COT数据: 按每份材料的四类目标条数补齐(总量随体量, 配比 20/30/20/30),
        每类用自己的真实样本 few-shot 与生成要求。existing = 模板通道已产出(用于算缺口)。"""
        if not self.llm_generator:
            return []

        samples = []
        source = os.path.basename(result.filepath)
        language = detect_language(text)

        # 按四类目标条数构造种子队列: 某类缺口 = 该类目标条数 - 模板通道该类已有条数
        have: Dict[str, int] = {}
        for sd in (existing or []):
            have[sd.get("task_type")] = have.get(sd.get("task_type"), 0) + 1
        targets = self._class_targets(text, have)
        queue: List = []
        for cls, target in targets.items():
            gap = target - have.get(cls, 0)
            if gap <= 0:
                continue
            for sd in self._seeds_for_class(text, result, cls, gap):
                queue.append((cls, sd))
        logger.info("COT 配额(按体量 %d 字→%d 条): 目标 %s | 模板已产 %s | 待生成 %s | %s",
                    len(text or ""), self._source_budget(text), targets, have or {},
                    {c: sum(1 for x, _ in queue if x == c) for c in {q[0] for q in queue}},
                    source)

        for cur_cls, inst in queue:
            ok = False
            cur_type = task_type or "解释"
            for attempt in range(max(1, self.llm_max_attempts)):
                inferred = None
                fail_reason = ""
                try:
                    # 链必须只基于样本 input 生成(数对齐/不引入 input 外的条目):
                    # context 用 inst["input"], 不用全文——LLM 数全文条目会让
                    # 链与 input 截断节选脱节(输出 8 条而 input 只有 3 条)
                    prompt = self._build_cot_prompt(
                        inst["instruction"],
                        inst["output"],
                        inst.get("input") or text[:800],
                        language,
                        cur_cls,      # 本条样本的类别(决定用哪类真实样本 few-shot)
                    )
                    response = self.llm_generator.generate(
                        prompt,
                        system_prompt="你是一个专业的推理链生成助手。请为问答对补充详细的推理步骤。"
                    )

                    inferred, cot = self._parse_cot_response(response)
                    if not cot:
                        fail_reason = "empty_cot"
                        raise ValueError(fail_reason)
                    # 截断兜底: 链与 output 都被 max_tokens 腰斩的风险;
                    # output 裁到完整句, 链裁到最近句号; 剩余过短 → 丢弃
                    from ..utils.validation import trim_incomplete_output
                    out_trimmed = trim_incomplete_output(inst["output"])
                    if out_trimmed is None:
                        fail_reason = "output_truncated"
                        raise ValueError(fail_reason)
                    chain_trimmed = trim_incomplete_output(cot, min_len=60)
                    if chain_trimmed is None:
                        fail_reason = "chain_truncated"
                        raise ValueError(fail_reason)
                    # 数字一致性硬查(防幻觉): output 中的数值必须能追溯到材料/推理链
                    material = inst.get("input", "") or text[:800]
                    if not self._consistency_check(out_trimmed, material, chain_trimmed,
                                           full_material=text):
                        fail_reason = "numeric_inconsistent"
                        raise ValueError(fail_reason)
                    # 类别 = 本样本所属的目标类别(不再取全篇预判, 否则一类吃满全库)
                    final_type = cur_cls
                    samples.append({
                        "instruction": inst["instruction"],
                        "input": inst.get("input", ""),
                        "chain_of_thought": chain_trimmed,
                        "output": out_trimmed,
                        "source": source,
                        "task_type": final_type,
                    })
                    ok = True
                except Exception as e:
                    if attempt == 0:
                        logger.warning("COT LLM 校验失败(%s), 丢弃: %s",
                                       fail_reason or e, str(inst.get("output"))[:40])
                if ok:
                    break
                cur_type = cur_cls
                # 拒绝采样(失败重试制): 只对"推理"类重试——算式重算/数字一致性
                # 是可机判的可靠信号, 重试直接换质量; 建议/比较/解释类无机判信号,
                # 重试与现有 LLM 语义抽检重叠, 维持单次生成
                if cur_type != "推理":
                    break
                if attempt > 0:
                    logger.info("COT 推理类重试第 %d 次", attempt + 1)

        return samples

    # 四类目标配比(= 规范 20/30/20/30)。只作**每份材料的目标比例**, 每份实际条数
    # 由材料体量决定: 长材料 >10 条、短材料 <10 条都正常, 不做固定条数截断,
    # 以免把高质量样本无端裁掉; 最终以全库比例落在 ±5% 内为准。
    _COT_RATIO = {"解释": 0.20, "建议": 0.30, "比较": 0.20, "推理": 0.30}
    _COT_CHARS_PER_SAMPLE = 500   # 材料每 ~500 字出 1 条样本
    _COT_MIN_PER_SOURCE = 4       # 下限: 四类各至少 1 条
    _COT_MAX_PER_SOURCE = 30   # 上限(超长文按 30 条: 76k 字按 500 字/条会算到 150 条, 过重)      # 上限: 防超长材料打爆调用量

    def _source_budget(self, text: str) -> int:
        """每份材料的样本总量: 由体量决定(每 ~500 字 1 条), 夹在 [4, 40] 之间。"""
        n = int(round(len(text or "") / float(self._COT_CHARS_PER_SAMPLE)))
        return max(self._COT_MIN_PER_SOURCE,
                   min(self._COT_MAX_PER_SOURCE, n))

    def _class_targets(self, text: str, have: Dict[str, int] = None) -> Dict[str, int]:
        """每份材料的四类目标条数。总量先按体量定; 若模板通道已产出的某类超出该配比
        (如清点模板一次出 3 条推理), 就把总量抬到让其它类也能按比例跟上的程度 ——
        每份都回到 20/30/20/30, 且**不删除任何已产出的样本**。
        """
        have = have or {}
        n = self._source_budget(text)
        for cls, r in self._COT_RATIO.items():
            h = have.get(cls, 0)
            if r > 0 and h > 0:
                n = max(n, int(h / float(r) + 0.999))      # ceil(h / 配比)
        n = max(self._COT_MIN_PER_SOURCE, min(self._COT_MAX_PER_SOURCE, n))
        return {cls: max(1, int(round(n * r))) for cls, r in self._COT_RATIO.items()}

    # 按类生成要求与 few-shot（真实样本风格压缩自 HF 数据集，与 skill 模板同源）
    _COT_REQ = {
        "推理": "每步写明算式/依据，中间值可验证，最终结论与回答一致；禁止跳步、禁止只给答案。",
        "解释": "因果链清晰，机制有依据，不编造；每步给出理由，结论与回答一致。",
        "建议": "先明确目标与约束，再列出候选方案，给出取舍理由与预期效果，最终收敛为建议结论。",
        "比较": "明确比较维度，逐项事实对比，给出落在具体情境的结论，不能只写“各有优劣”。",
    }
    # skill cot-dataset-generation 第二步的**真实数据集模板** —— 原样嵌入, 不做改写。
    # 每类 = (数据集, 真实样本 JSON 原文, 中文释义[供中文材料参考])
    _COT_SHOT = {
        "推理": ("openai/gsm8k", '{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"}',
                 "逐句写出算式与中间值(<<算式=结果>> 形式), 最后收敛到最终答案(#### 答案)"),
        "解释": ("allenai/sciq", '{"question": "What type of organism is commonly used in preparation of foods such as cheese and yogurt?", "correct_answer": "mesophilic organisms", "support": "Mesophiles grow best in moderate temperature, typically between 25°C and 40°C (77°F and 104°F). Mesophiles are often found living in or on the bodies of humans or other animals. The optimal growth temperature of many pathogenic mesophiles is 37°C (98°F), the normal human body temperature. Mesophilic organisms have important uses in food preparation, including cheese, yogurt, beer and wine."}',
                 "question/correct_answer/support 三字段: 先给答案, 再用 support 说明机制与依据"),
        "建议": ("jonathanli/legal-advice-reddit", '{"title": "My former employer fired everyone when they had to shutdown so they could avoid paying out sick time... Is this legal?", "body": "The title says most of it. I just heard back from a friend... Cafe Mooby stayed open until the last minute, and then terminated everyone\'s employment... I\'m just curious if it\'s legal, and if the workers can file a wrongful termination class-action lawsuit or something?"}',
                 "真实样本为「情景 + 诉求」(title/body), 链要先列约束再对比方案后给可执行建议"),
        "比较": ("CarperAI/openai_summarize_comparisons", '{"prompt": "SUBREDDIT: r/relationships\nTITLE: To admit or not to admit snooping...\nPOST: I [25M] have snooped in the past...", "chosen": "TL;DR: Snooped, found something, should I admit what I found...", "rejected": "TL;DR: I snooped, we talked about it, she wants to work it out, I\'m not sure..."}',
                 "真实样本给出 prompt 与 chosen/rejected 两个候选, 链要按维度逐项对比后下结论"),
    }

    # skill cot-dataset-generation: 模板未命中 ≠ 不能生成 —— 问答对由材料
    # **抽取式**构造(每类固定问法 + 材料节选 + 材料首句作答案), 之后走同一条
    # LLM 链与同一套硬校验。中英各一套问法(英文形态此前缺失, 导致英文散文/
    # 扫描件整份零产出)。
    _SEED_ASK = {
        "推理": ("根据以下材料，清点其中总共列出了几个条目？请逐条核对并给出总数。",
                 "Based on the following material, how many items are enumerated "
                 "in total? Check each item and give the count."),
        "解释": ("根据以下材料，解释其中涉及的概念或机制。",
                 "Based on the following material, explain the concept or "
                 "mechanism it describes."),
        "建议": ("根据以下材料，提出具体可操作的建议。",
                 "Based on the following material, give concrete actionable "
                 "recommendations."),
        "比较": ("根据以下材料，比较其中提到的对象，说明差异与适用情境。",
                 "Based on the following material, compare the objects mentioned, "
                 "explaining the differences and when each applies."),
    }

    @staticmethod
    def _first_excerpt(text: str, limit: int) -> str:
        """取材料开头到最近句末的一段（无句末标点时按 limit 截断）。"""
        t = (text or "").strip()
        if len(t) <= limit:
            return t
        head = t[:limit]
        cut = max(head.rfind(c) for c in "。！？!?.")
        return head[:cut + 1] if cut >= 0 else head

    def _extractive_seed(self, text: str, language: str,
                         task_type: Optional[str] = None) -> Optional[Dict]:
        """抽取式问答对（模板通道未命中时的兜底）：问法按类别固定，答案取材料
        首句、材料节选作 input —— 全部来自材料原文，不引入任何生成内容。"""
        t = (text or "").strip()
        if len(t) < 30:
            return None
        tt = task_type if task_type in self._SEED_ASK else "解释"
        ask_zh, ask_en = self._SEED_ASK[tt]
        return {
            "instruction": ask_zh if language == "zh" else ask_en,
            "input": self._first_excerpt(t, 600),
            "output": self._first_excerpt(t, 200),
        }

    # 句子切分(句末标点或换行) —— 抽取式锚点的最小单位
    _SENT_SPLIT = re.compile(r'(?<=[。！？!?；;])\s*|\n+')
    _SENT_END_CHARS = "。！？!?…"
    # 编号条目(含中文序号与 ASCII), 供"措施/清单"类锚点使用
    _ITEM_RE = re.compile(r'(?:^|\n)\s*(?:[（(]?[一二三四五六七八九十]{1,3}[）)、.．]|'
                          r'\d{1,3}[.)])\s*(.{8,200})')

    @staticmethod
    def _is_complete(s: str) -> bool:
        """完整句判定: 中文句末标点收尾, 或拉丁尾部以句点收尾。
        标题句(以"："收尾)不算 —— 拿它当答案会被截断校验整条丢弃。"""
        y = (s or "").strip()
        if not y:
            return False
        if y[-1] in COTConverter._SENT_END_CHARS:
            return True
        return y[-1] == "." and not re.search(r'[\u4e00-\u9fff]', y[-30:])

    def _class_anchors(self, text: str, cls: str, k: int) -> List[str]:
        """为某一类找 k 个**不同**且**完整句**的材料落点(句子/条目/对象对):
        同类多条的答案与 input 取自不同落点, 避免同材料同问重复。"""
        t = text or ""
        sents = [x.strip() for x in self._SENT_SPLIT.split(t) if x and len(x.strip()) >= 12]
        pat = {"解释": self._HINT_EXPLAIN, "建议": self._HINT_ADVICE,
               "比较": self._HINT_COMPARE, "推理": self._HINT_MATH}.get(cls)
        picked, seen = [], set()

        def take(x):
            x = x.strip()
            if not x or not COTConverter._is_complete(x):
                return False
            key = x[:40]
            if key in seen:
                return False
            seen.add(key)
            picked.append(x)
            return True

        for sx in sents:
            if pat is not None and not pat.search(sx):
                continue
            take(sx)
            if len(picked) >= k:
                return picked[:k]
        items = [m.group(1).strip() for m in self._ITEM_RE.finditer(t)]
        if cls == "比较":
            # 比较类: 编号条目两两成对("A 与 B"), 材料中并列的措施/对象即可对比
            pairs = []
            for i in range(0, max(0, len(items) - 1), 2):
                pair = "%s 与 %s" % (items[i].rstrip("。"), items[i + 1])
                pairs.append(pair[:200])
                if len(pairs) >= k:
                    break
            if pairs:
                return pairs
        for it in items:
            take(it)
            if len(picked) >= k:
                return picked[:k]
        for sx in sents:          # 兜底: 完整长句(保证每类都能出到目标条数)
            if len(sx) < 20:
                continue
            take(sx)
            if len(picked) >= k:
                break
        return picked[:k]

    @staticmethod
    def _excerpt_around(text: str, anchor: str, limit: int) -> str:
        """取锚点前后各半的窗口, 保证锚点本身在 input 里可见(链只能引用 input 内内容)。"""
        t = text or ""
        i = t.find(anchor)
        if i < 0:
            return COTConverter._first_excerpt(t, limit)
        half = max(120, limit // 2)
        return COTConverter._first_excerpt(t[max(0, i - half):], limit)

    def _seeds_for_class(self, text: str, result: ReaderResult, cls: str,
                         k: int) -> List[Dict]:
        """某一类的 k 个抽取式种子: 该类固定问法 + 材料锚点作答案 + 锚点上下文作 input。
        答案必须是**完整句**(否则会被截断校验丢弃); 锚点不足时用材料不同窗口兜底,
        全部取自材料原文, 不引入生成内容。"""
        ask_zh, ask_en = self._SEED_ASK[cls]
        language = detect_language(text)
        ask = ask_zh if language == "zh" else ask_en
        seeds: List[Dict] = []
        for a in self._class_anchors(text, cls, max(k * 2, k + 3)):
            if len(seeds) >= k:
                break
            out = self._first_excerpt(a, 200)
            if not COTConverter._is_complete(out):
                continue
            seeds.append({
                "instruction": "%s（围绕材料中的「%s」）" % (ask, a.strip()[:24]),
                "input": self._excerpt_around(text, a, 800),
                "output": out,
            })
        i = 0
        while len(seeds) < k and i < k * 3:      # 兜底: 材料不同窗口
            win = (text or "")[i * 300:]
            out = self._first_excerpt(win, 200)
            if len(out) >= 20 and COTConverter._is_complete(out):
                seeds.append({"instruction": ask,
                              "input": self._first_excerpt(win, 800),
                              "output": out})
            i += 1
        return seeds

    def _build_cot_prompt(self, instruction: str, output: str,
                          context: str, language: str,
                          task_type: Optional[str] = None) -> str:
        """构建COT生成prompt（单模板参数化 + 按类 few-shot，与 skill 模板同构）"""
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
                      if task_type else
                      "\n请先判断这条材料最适合的推理类别（解释/推理/建议/比较），"
                      "在推理过程前单独一行输出“类别: X”，再按该类别的风格生成推理过程。")

        if language == "zh":
            return f"""请根据以下问答对，生成详细的推理过程(chain-of-thought)。

原始文本的上下文（供参考）：\n{context[:1200]}

问答对：\n指令：{instruction}\n回答：{output}\n{class_line}\n{shot_line}
纪律（必须遵守）：
- 推理过程的每一步只能引用上方上下文/问答对中**明确出现**的内容；
- 未在上方文本中出现的事实、数字、专名、条目一律禁止引入（禁止凭常识或训练记忆补充政策细节）；
- 若信息不足，只基于已给信息推理，禁止外推。
请按以下格式输出（只输出推理过程本身，不要额外解释）：\n1. ...\n2. ...\n3. ...\n因此结论是：{output}"""
        else:
            return f"""Generate a detailed chain-of-thought reasoning process for the following Q&A pair.

Context:
{context[:800]}

Q&A:
Instruction: {instruction}
Answer: {output}
Task category: {task_type or "self-judge (explanation/reasoning/advice/comparison)"}
Requirements for this category:
{req}
{shot_line}
Output format (reasoning process only):
1. ...
2. ...
3. ...
Therefore, the answer is: {output}"""

    def _parse_cot_response(self, response: str):
        """解析COT输出：提取可选的任务类别前缀与推理链文本。
        返回 (task_type or None, 链文本 or None)。"""
        text = response.strip()
        if len(text) < 10:
            return None, None
        m = re.match(r'^类别\s*[:：]?\s*(解释|推理|建议|比较)', text)
        if m:
            return m.group(1), text[m.end():].strip()
        return None, text