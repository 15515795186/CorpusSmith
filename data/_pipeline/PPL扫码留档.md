# PPL 扫码留档(2026-09-25)

PPL(困惑度)闸门的标定过程、决策依据与分布, 供后续复查/复现。

## 一、阈值结论(当前生效)

`enable_ppl_filter=True`, **`ppl_threshold=25.0`(极端异常闸)**, `ppl_quantile=0.95`(仅扫码标定用), `ppl_cache_dir=D:/model`。

**为什么不用分位阈值**: q95 在两次语料变化下分别给出 16.21 / 16.94, 但每次都切到正常语料:

| 阈值 | 淘汰清单 | 问题 |
|---|---|---|
| q95=16.94(全量 80, 最新分布) | 25.0 scan_en_03_knowledge_transitions.pdf, 24.0 scan_en_09_exploring_the_extreme.pdf, 19.9 word_05_e775227929298780ac6f.docx, 18.9 zh_pdf_06_石油天然气十五五规划.pdf | 4 份**全是正常内容**, 含 2 份刚重爬的英文正文页 |
| 25.0(当前) | **0 份** | 只拦机器翻译腔/乱码级异常 |

实测数据(全量 80 份): 最低 2.12 / 次低 3.09(都是高度公式化的政务方案), 高分端全是技术页、表格页、版式页 ——
**语料池里不存在"机器翻译腔/乱码"异常簇**, 因此分位阈值只会在正常内容上做减法。PPL 的定位是"花 API 钱之前的免费预筛", 内容质量判断交给转换后硬校验、LLM 评分与人工仲裁(那些机制已经存在)。

补充证据(短页天然偏高): 字符 1000~3000 段 q95=24.01, ≥3000 字符段 q95=16.21; 长度与 PPL 弱负相关(r=-0.107)。
本轮重爬的 `版式pdf英文版` 是**单页** OCR 文本(1.6k~3.5k 字符 ≈ 500~800 token), 天生落在高 PPL 一侧。

## 二、全量分布(n=80 = 文字层 50 + OCR 30)

| 口径 | n | min | p50 | p75 | p90 | p95 | max |
|---|---|---|---|---|---|---|---|
| 全量 80 | 80 | 2.12 | 7.75 | 10.58 | 14.63 | 16.94 | 24.98 |

按文本长度分层:

| 字符数 | n | min | p50 | p90 | p95 | max |
|---|---|---|---|---|---|---|
| 0~1000 | 5 | 3.09 | 5.08 | 12.12 | 12.12 | 12.12 |
| 1000~3000 | 28 | 2.53 | 7.63 | 19.86 | 24.01 | 24.98 |
| >=3000 | 47 | 2.12 | 7.88 | 14.63 | 16.21 | 18.93 |

每类语料分位:

| 分类 | n | p50 | p90 | p95 | max |
|---|---|---|---|---|---|
| txt | 10 | 6.01 | 8.31 | 9.30 | 9.30 |
| word | 10 | 9.90 | 13.39 | 19.86 | 19.86 |
| 截图html | 10 | 5.99 | 10.58 | 13.78 | 13.78 |
| 文本pdf中文版 | 10 | 8.64 | 13.73 | 18.93 | 18.93 |
| 文本pdf英文版 | 10 | 9.36 | 14.63 | 16.21 | 16.21 |
| 源码html | 10 | 6.77 | 9.71 | 9.78 | 9.78 |
| 版式pdf中文版 | 10 | 6.15 | 8.19 | 8.93 | 8.93 |
| 版式pdf英文版 | 10 | 14.41 | 24.01 | 24.98 | 24.98 |

## 三、版式pdf英文版 重爬记录(旧 10 份作废)

旧 10 份实为 Project Gutenberg **下载/封面页**(OCR 后只有书名+下载链接, 767~1843 字符), 已移出项目目录(备份 `/d/hermes/cache/scratch/_old_scan_en_gutenberg`)。新 10 份取自美国政府/机构公开文档(public domain), 按 COT 四类配比选材, 每份 1 页、无文字层、需 OCR。

| 文件 | 机构 | COT 类别 | 源页 | 源文档页数 | 源页选取 |
|---|---|---|---|---|---|
| scan_en_01_mwee_guide.pdf | NOAA | 解释 | 第6页 | 74 | first_content_page(front_section) |
| scan_en_02_meteorology_educator.pdf | NASA GPM | 解释 | 第9页 | 146 | first_content_page(front_section) |
| scan_en_03_knowledge_transitions.pdf | NASA | 建议 | 第3页 | 7 | first_content_page(front_section) |
| scan_en_04_mariners_guide.pdf | NOAA NHC | 建议 | 第1页 | 2 | first_text_page |
| scan_en_05_om_best_practices.pdf | DOE FEMP | 建议 | 第5页 | 321 | first_content_page |
| scan_en_06_comparing_instruments.pdf | NIST | 比较 | 第7页 | 57 | first_content_page(front_section) |
| scan_en_07_hardness_five_methods.pdf | NIST | 比较 | 第6页 | 29 | first_content_page(front_section) |
| scan_en_08_flight_control_math.pdf | NASA | 推理 | 第3页 | 12 | first_content_page(front_section) |
| scan_en_09_exploring_the_extreme.pdf | NASA | 推理 | 第7页 | 88 | first_content_page |
| scan_en_10_thermocouple_methods.pdf | NIST | 推理 | 第3页 | 36 | first_content_page(front_section) |

**正文页选取规则的演进(踩坑记录, 都实测过)**

1. "第一个 >= 300 字符的页" -> 选中**封面**(标题+作者列表);
2. 加" >=1200 字符 且 >=5 个句末标点" -> 仍选中**目录页**("1. INTRODUCTION .. 3");
3. 改成"全局 body_score 最高页" -> 跑到**参考文献/附录**(第 65/91/243 页);
4. 最终: **顺序扫描**(优先前 15 页) + `is_front_matter` 否决(Contents / List of Tables / 目录点线 / 行首编号 / 全短行) + (字符数 >= 800, 句末标点 >= 8) 双门槛; 未果才退化为 body_score 择优 -> 文字最多页 -> 第 3 页。

另一处踩坑: 验证时把 **PDF 直接喂给 Qwen-VL OCR reader** 会报 `The image format is illegal`(reader 只收图片, mimetype 变成 application/pdf) —— 必须先渲染成 PNG 再 OCR, 与流水线 PDF 路径一致。

复核: 10/10 份 OCR 后逐份看过开头, 均为连续正文(指南框架/使用说明/步骤建议/仪器比对/实验目的), 非封面或目录。

## 四、口径与复现

- **清洗必须必过**: txt/docx/文字层 PDF/OCR 文本 一律过 `TextCleaner`, html 先过 `HTMLCleaner`。早前版本只在 `source=="ocr"` 时清洗, 文字层 PDF 拿"未清洗原文"打分(20 份系统性偏低, 最大 -2.84); 已修, 修后文字层 50 份与首次扫码**逐值一致(偏差 0.0)**。
- 分位取法统一 `ceil(n*q)-1`, 与 `PPLFilter.calibrate_threshold` 同源(早期 `int(n*q)` 导致分位表与建议值自相矛盾)。
- 模型 `Qwen/Qwen2.5-0.5B`, 缓存根 `D:/model`(离线 `HF_HUB_OFFLINE=1`, 显式 `cache_dir` 透传); 单文本只评前 2048 token。
- OCR 文本缓存 `data/_pipeline/ocr_cache/`(40 份), 重跑零 API 成本。
- 仪表盘: `ppl_scan_scores_full80.json`(逐文件分数)、`ppl_scan_with_length.json`(含文本长度)、`ppl_scan_summary.json`。

```bash
# 全量扫码(走 OCR 缓存, 零 API)
D:/Anaconda3/envs/Unlimited-OCR/python.exe data/_pipeline/scan_ppl_distribution.py
# 只跑文字层语料 / 只跑需 OCR 的语料
D:/Anaconda3/envs/Unlimited-OCR/python.exe data/_pipeline/scan_ppl_distribution.py --no-ocr
D:/Anaconda3/envs/Unlimited-OCR/python.exe data/_pipeline/scan_ppl_distribution.py --only-ocr
# 重爬英文版式 PDF(下载 + 取正文页 + 单页无文字层渲染); 定点重做某份
D:/Anaconda3/envs/Unlimited-OCR/python.exe data/_pipeline/rebuild_scan_en_pdf.py
D:/Anaconda3/envs/Unlimited-OCR/python.exe data/_pipeline/rebuild_scan_en_pdf.py --redo <slug>
```

## 五、未决

- PPL 闸门目前对语料池"零淘汰"; 建议后续用全量跑的产出与 LLM 评分对照, 再决定是否保留此闸门。
- 只有 `版式pdf英文版` 的 10 份语料纳入了版本库(其余 70 份仍在被忽略的 data/ 下)。

## 六、PPL 闸门的边界(实测, 别指望它包打天下)

| 样本形态 | 实测 PPL | PPL 闸门是否拦 |
|---|---|---|
| 乱码/随机拉丁串 | **549.30** | 拦得住 |
| 单页版式英文 OCR 正文(短页) | 16.0~25.0 | 分位阈值会误拦, 25 的绝对闸放行 |
| 政务规划正文(专名/页码密集) | 17.2~19.9 | 同上 |
| 重复退化文本(同一句 ×12) | **1.82** | **拦不住**(高度可预测 → PPL 低) |
| 中英混插模板句(×25) | **1.23** | **拦不住** |

结论: **PPL 高 = 难预测**, 所以它能拦乱码/机翻腔(不可预测), 但**拦不住重复退化文本**(可预测)。
重复/退化由 `quality_filter` 的 `max_repeat_char_ratio` / `garbage_ratio_threshold` 负责, 两者互补, 不要指望 PPL 覆盖。
