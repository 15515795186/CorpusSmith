"""
零依赖 .env 加载器

用法：项目根放置 `.env`（已被 .gitignore 覆盖），内容如：

    DASHSCOPE_API_KEY=sk-xxxxx

规则：
- 真实环境变量优先于 .env（临时调试的 shell 变量不会被静默覆盖）；
- 支持 `#` 注释、`export KEY=VALUE` 前缀、单/双引号包裹的值；
- 只填充"未设置或为空"的键；
- 任何情况下不把值写进日志。

安全提示：Windows 下可对 .env 右键 → 安全 → 高级，移除其他账户的读取权限。
"""

import os
from typing import Dict, List, Optional

# 项目根 = 本文件所在包 (file_processor/) 的上一级
DEFAULT_ENV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def load_dotenv(path: Optional[str] = None) -> Dict[str, bool]:
    """把 .env 中的键值注入 os.environ（已存在的真实变量不覆盖）

    Returns: {key: True} —— 仅键名，绝不返回/打印值本身。
    文件不存在时返回空 dict，不报错（.env 是可选项）。
    """
    if path is None:
        path = DEFAULT_ENV_PATH
    loaded: Dict[str, bool] = {}
    if not os.path.isfile(path):
        return loaded

    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
    except OSError:
        return loaded

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("\"", "'"):
            value = value[1:-1]
        if not key:
            continue
        if not os.environ.get(key):
            os.environ[key] = value
        loaded[key] = True
    return loaded


def env_keys_present() -> List[str]:
    """当前进程可见的凭据键名（不含值），供启动日志提示来源用"""
    return [k for k in ("DASHSCOPE_API_KEY", "OPENAI_API_KEY") if os.environ.get(k)]
