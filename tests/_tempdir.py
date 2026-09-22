"""测试用临时目录工具

两个环境约束：
1. 沙箱环境下系统临时目录（%TEMP%）可能不可写，统一在工作区内创建；
2. 某些文件沙箱拒绝写入由 ``tempfile.mkdtemp`` 创建的目录（mode 0700），
   因此改用 ``os.makedirs`` + 唯一名称。
"""

import os
import shutil
import uuid

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_TMP = os.path.join(PROJECT_ROOT, ".test_tmp")


def workspace_tempdir() -> str:
    """在工作区 .test_tmp/ 下创建一个唯一临时目录"""
    path = os.path.join(BASE_TMP, f"case_{uuid.uuid4().hex[:12]}")
    os.makedirs(path, exist_ok=True)
    return path


def cleanup(path: str):
    """删除临时目录（忽略失败，例如文件被占用）"""
    shutil.rmtree(path, ignore_errors=True)