# converters/ — 数据集转换层

把清洗+质检过关的文本转换成三种训练样本。**规则提取为主，LLM 生成是可选兜底**
（默认 `enable_llm=False`，因此当前产出 100% 确定性、可复现）。

| 文件 | 产出格式 | 触发 LLM 的条件 |
|------|----------|----------------|
| `pretrain_converter.py` | `{"text", "meta"}` | 从不（纯分块） |
| `instruction_converter.py` | Alpaca `{instruction, input, output, source}` | 模板样本 < 20 且传入 llm_generator |
| `cot_converter.py` | Alpaca + `chain_of_thought` | 模板样本 < 10 且传入 llm_generator |

## ⚠️ input 字段语义（开卷/闭卷契约——训练前必须理解的一条规则）

**答案内容来自某篇具体文档的样本（阅读、列举、步骤），来源材料必须放进 `input`。**
否则等于要求模型闭卷背诵一篇题目里没给的文档——这种数据训练出来的行为就是幻觉。

- 合规留空（`input: ""`）的只有两类：
  - 定义式通识问答（"什么是X？"题面自包含，不指向特定文档）
  - 因果/条件推理（已知条件已经写进题面的"因为…所以…"句式）
- 本包所有守卫（材料预算截断、计数与材料一致、要点从截断后材料抽取）
  都服务于同一原则：**答案不得引用 input 里看不到的内容**。
- `verify_datasets.py` 只查字段存在性，查不出这种语义错误；
  契约由 `tests/test_converters.py` 的 11 个契约测试锁死。

## pretrain_converter.py

- `chunk_size<=0` 时整文档输出一条；否则用 langchain
  `RecursiveCharacterTextSplitter` 按 `["\n\n","\n","。",".","！","？"," ",""]`
  优先级切块（中文句号优先于英文句点）。
- 防御：langchain 要求 overlap<chunk_size，旧版会直接 ValueError 崩掉整个 pipeline；
  现在自动收敛（`--chunk-size 100` 也能跑）。
- 每块独立重新检测语言；`<10` 字的碎块丢弃；meta 记
  `source/type/language/chunk_index/chunk_total/size_bytes/token_count_est`。

## instruction_converter.py — 四种样本形态

`_generate_from_templates()` 按序尝试（每文档上限 30 条）：

1. **标题阅读**（Markdown `#`~`###` 标题）：标题后的正文段（≥60 字）作为材料进
   `input`（截 1200），`_key_points()` 从**截断后的材料**里抽每段首句组成编号要点当
   `output`——保证"开卷答题"，要点绝不引用被截掉的文字。
   纯编号标题（"第四条"、"（三）"）不带语义，改用无指代题面；英文标题走英文措辞。
2. **符号列表提取**（`-`/`*` 行 ≥3 条）：条目进 `input`，`output` 为编号整理结果；
   条目按材料预算截断，保证输出条数=材料可见条数。上下文标题只用于题面指代，
   找不到就不硬造"相关内容"这种空话。
3. **编号条目列举**（`一、/1./（三）/第X条` 起始的行块 ≥3 条）：政策白皮书式并行
   措施的正确落点是"根据材料逐条列举并编号"（这是提取任务），**不是** COT
   "逐步说明完整过程"（并行措施没有时序，那样生成属于语义错配）。
   相邻条目间只允许空白（保证同一列举块）；描述中位长 <10 的块是目录/章节名，拒绝。
4. **定义式抽取**（`X是指/即/属于/意为Y` 行首匹配）：唯一合法的闭卷留空形态，
   但有严格清洗门控 `_definition_sample()`：
   - 术语剥 `**加粗**`、剥列表/章节编号前缀、限长 16 字、拒动词误接结尾（"仍然是"）；
   - 释义**截到首个句末标点**——旧正则一口气吞 200 字，把"…图件。第十八条"的
     下一条编号都写进答案；
   - 拦截"是指是指"二次触发。
   （历史 bug：旧版此处 output 恒为 None 的模板是死代码，标题样本实际从未生成。）

LLM 兜底 `_generate_with_llm()`：取前 5 块 ×1500 字构造中文/英文提示词（按文档语言），
few-shot 明确要求"题目需要参考材料的，材料节选进 input；自包含通识题 input 留空"。
解析容忍 ```json 包裹与裸数组。

## cot_converter.py — 三种推理形态

`_extract_step_cot()`：
1. 三套编号步骤正则（`1.`/`第X步`/`步骤一`），然后过 **三重守卫
   `_real_process_steps()`**：
   - 纯数字序列**按 +1 严格递增切段取最长段**（`1,2,3,1,2,3` 合并编号和
     `3./3./1.` 扫描版章节碎片都是目录噪声，不是同一个序列）；
   - 步数 2~12、描述中位长度 ≥10 字（短词占多数的是章节名不是步骤）；
   - 英文论文节名（Introduction/Method/Results…）过半即拒。
2. **条目清点样本**（所有通过守卫的块）：数数不预设时序，对清单和流程都成立。
   `chain_of_thought` 逐条列出后计数，`output` 计数与 `input` 可见条目严格一致
   （材料预算外的条目连同计数一并舍弃）。
3. **流程逐步样本**（仅强时序证据时追加）：≥2 个步骤文本含"首先/然后/最后/第X步"，
   或块前 80 字有"流程/步骤如下/办理程序"。宁可少生成，也不对着一列公司名讲"完整流程"。

`_extract_definition_cot()`：
- "因为/如果 P 所以 Q"两组式 → 题面自包含，`input` 合规留空；
- "首先P其次Q最后R"三组式 → 原句是材料，进 `input`（旧版题面"请逐步分析完整过程"
  不指代任何内容，是无锚点指令）。

LLM 兜底：对已有模板样本补写 `chain_of_thought`（继承原 `input`），
`_parse_cot_output()` 剥 ``` 围栏与"推理过程："前缀。

## 依赖

- 内部：`readers.base.ReaderResult`（入参）、`utils.lang_detect`
  （中英分流、`contains_english` 决定标题模板语言）
- 可选：`llm/` 的生成器实例由 pipeline 注入（构造参数 `llm_generator`）
- 被 `pipeline._convert_and_output()` 与 `rebuild_derived.py` 引用；
  产出的记录由 `formatters/jsonl_formatter.py` 做样本级去重后落盘
