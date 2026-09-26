"""样本级校验公共函数（转换后校验层）。

数字一致性是防幻觉的强证据硬查：output 中出现的数值必须能在
材料/推理链中找到出处，否则视为编造或答非所问。
"""

import re

_NUM_RE = re.compile(r'\d+(?:\.\d+)?%?')
_FULLWIDTH_DIGITS = str.maketrans('０１２３４５６７８９', '0123456789')
# 要点化列举的行首编号("1. " "(2) " "（3）" "4、"), 不属于正文数值,
# 提取 output 数字前剥离, 否则列举题被编号数字误杀
_BULLET_NUM_RE = re.compile(r'^\s*[（(]?[0-9]{1,3}[）)、.．]\s*', re.MULTILINE)


def strip_bullet_numbers(text: str) -> str:
    """剥离行首编号标记(仅阿拉伯数字编号, "2.5%" 等小数不受影响)。"""
    if not text:
        return ''
    return _BULLET_NUM_RE.sub('', text)


def numbers_in(text: str) -> set:
    """提取文本中的数值 token（全角数字/百分号归一为半角）。"""
    if not text:
        return set()
    return set(_NUM_RE.findall(
        text.translate(_FULLWIDTH_DIGITS).replace('％', '%')))


def numeric_consistency(output: str, material: str, chain: str = '',
                        full_material: str = '') -> bool:
    """数字一致性硬查。

    - output 为空或过短 → 不合格（没有训练价值）；
    - output 无数值 → 无法程序化断言，放行交 LLM 语义抽检；
    - output 有数值 → 必须 ⊆ 材料∪推理链 的数值集合，否则判为幻觉。
    - 行首编号("1. " "(2)")剥离后不参与判定(要点化列举不误杀);
    - full_material: 指代题 input 是限长节选, 节选内未检出时用全文复查一次,
      覆盖"兜底节选截断恰好漏掉答案数字"的假阳性。
    """
    if not output or len(output) < 2:
        return False
    out_nums = numbers_in(strip_bullet_numbers(output)) - {'0', '1'}  # 0/1 常见于量词/序号, 不作为证据
    if not out_nums:
        return True
    ctx = numbers_in(material) | numbers_in(chain)
    if out_nums <= ctx:
        return True
    if full_material and out_nums <= (numbers_in(full_material) | numbers_in(chain)):
        return True
    return False


# ============ 数学题重算（《数据集介绍.md》6.3：计算已验证）============
_MATH_EQ = re.compile(r'(\d+(?:\.\d+)?)\s*([+\-×xX*÷/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)')
_OPS = {'+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '×': lambda a, b: a * b, 'x': lambda a, b: a * b,
        'X': lambda a, b: a * b, '*': lambda a, b: a * b,
        '÷': lambda a, b: a / b, '/': lambda a, b: a / b}


def math_chain_check(chain: str):
    """重算推理链中的显式等式（a op b = c）。

    - 无等式 → None（跳过，交语义抽检）；
    - 任一条等式验算错误 → False（剔除）；
    - 全部等式正确 → True。
    """
    if not chain:
        return None
    eqs = _MATH_EQ.findall(chain)
    if not eqs:
        return None
    for a, op, b, c in eqs:
        try:
            if abs(_OPS[op](float(a), float(b)) - float(c)) > 1e-6:
                return False
        except (ZeroDivisionError, ValueError, OverflowError):
            return False
    return True


# ============ 隐私脱敏扫描（《数据集介绍.md》1.3：禁止保留隐私信息）============
_PHONE = re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)')
_ID_LEN = re.compile(r'(?<!\d)\d{13,19}(?!\d)')


def privacy_scan(text: str) -> set:
    """扫描文本中的个人隐私模式。

    返回命中的类型集合：phone（手机号）/ idcard（18 位身份证）/ bankcard（银行卡 13-19 位）。
    """
    if not text:
        return set()
    hits = set()
    if _PHONE.search(text):
        hits.add('phone')
    for grp in _ID_LEN.findall(text):
        if len(grp) == 18 and (grp[-1] == 'X' or grp[-1] == 'x'
                               or re.fullmatch(r'\d{17}[\dXx]', grp)):
            hits.add('idcard')
        else:
            hits.add('bankcard')
    return hits


# ============ 代码题运行（《数据集介绍.md》6.3：代码已验证，默认关闭）============
import subprocess  # noqa: E402
import sys  # noqa: E402

_CODE_BLOCK = re.compile(r'```(?:python|py)\n(.*?)```', re.DOTALL)
_UNSAFE_CODE = re.compile(r'^\s*(import|from)\s+(os|subprocess|socket|requests|urllib|'
                          r'sys|shutil|pathlib|ctypes|http|ftplib|telnetlib|importlib)',
                          re.MULTILINE | re.IGNORECASE)
_UNSAFE_CALLS = re.compile(r'__[^_].*__|exec\s*\(|eval\s*\(|open\s*\(|input\s*\(')


def code_exec_check(text: str, expected_output: str = ''):
    """提取文本中 python 代码块并安全运行，验证输出与预期一致。

    - 无代码块/代码不安全 → None（跳过）；
    - 运行失败/超时/输出不一致 → False；
    - 输出一致 → True。

    安全门槛：拒绝 import 危险模块、拒绝 dunder/exec/eval/open/input。
    此功能默认关闭（converter 开关 verify_code_exec），语料可能含恶意代码。
    """
    m = _CODE_BLOCK.search(text)
    if not m:
        return None
    code = m.group(1)
    if _UNSAFE_CODE.search(code) or _UNSAFE_CALLS.search(code):
        return None
    try:
        r = subprocess.run([sys.executable, '-c', code],
                           capture_output=True, text=True, timeout=3)
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    if r.returncode != 0:
        return False
    out = r.stdout.strip()
    if not expected_output.strip():
        return True if out else None
    return out == expected_output.strip()

# ============ LLM 输出截断兜底（max_tokens 截断防毒样本）============
_SENT_END = '。！？!?…'
_CJK_RE = re.compile(r'[\u4e00-\u9fff]')
# 拉丁文本(无汉字)把 "." 也算句末: 英文完整句以句点收尾, 若不认则
# 英文单句 output/推理链被判"截断"整条丢弃(实测踩到); 含汉字的中文
# 文本仍只用原集合, 免得在 "13.5" 这类数字处误裁。


def trim_incomplete_output(output: str, min_len: int = 30):
    """LLM 输出截断兜底。

    - 完整句结尾（句号/叹号/问号/省略号）→ 原样返回；
    - 多行要点式输出（规则 6 要点化产物，每行 ≤40 字，无句号约定）→
      只要清理尾部悬空编号行（"3"/"4." 这类被截的条目头），其余保留；
      这类输出以句号结尾属误杀，Propmpt 的要点化要求与之冲突；
    - 段落式半句结尾 → 裁剪到最近一个完整句号；
    - 裁剪后过短或无句号 → None（截断严重，交人工审核/丢弃）。
    """
    if not output:
        return None
    o = output.strip()
    # 句末判定: 中文标点一律认; "." 只在"尾部无汉字"时认 —— 英文完整句以句点
    # 收尾(中英混排的推理链尾部通常就是英文句), 而中文文本里的 "13." 不算句末。
    latin_tail = not _CJK_RE.search(o[-30:])
    if o[-1] in _SENT_END or (latin_tail and o[-1] == "."):
        return o
    lines = [l.strip() for l in o.split('\n') if l.strip()]
    if len(lines) >= 2 and max(len(l) for l in lines) <= 40:
        # 要点式：仅清尾部悬空编号行（纯编号/编号+≤4字残头）
        while lines:
            last = lines[-1]
            if re.fullmatch(r'[（(]?\d+[）).、．]?', last):
                lines.pop()
                continue
            if re.match(r'^\d+[.、．]', last) and len(last) <= 5:
                lines.pop()
                continue
            break
        out = '\n'.join(lines).strip()
        return out if out else None
    ends = _SENT_END + ("." if not _CJK_RE.search(o) else "")
    cut = max((pos for pos in (o.rfind(c) for c in ends) if pos >= 0), default=-1)
    if cut >= min_len:
        return o[:cut + 1]
    return None
