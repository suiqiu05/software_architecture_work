"""
bootstrap.py —— 所有入口共用的启动预检（可移植性基石）。

职责：
1. 把项目根目录注入 `sys.path`，保证从任意工作目录调用、以及进程池
   spawn 出的子进程都能 import 本地模块；
2. 检查 numpy / matplotlib 是否可用，缺失时打印**可执行的修复指引**，
   而不是抛一段与业务无关的 ImportError traceback；
3. 导出 `HERE`，各入口据此推导结果目录 —— 项目内禁止出现绝对路径。

用法（必须在 import 任何本地模块之前调用）：
    from bootstrap import HERE, ensure_ready
    ensure_ready(entry="main_exp.py")
"""

from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# 运行期必需的第三方包（其余全部使用标准库）
REQUIRED: tuple = ("numpy", "matplotlib")


def ensure_ready(extra: tuple = (), entry: str = "this script") -> str:
    """检查依赖；缺失时打印指引并 exit(2)，否则返回项目根目录。"""
    wanted = tuple(REQUIRED) + tuple(extra)
    missing = [m for m in wanted if importlib.util.find_spec(m) is None]
    if not missing:
        return HERE

    print("=" * 72)
    print("[FATAL] Missing required packages: " + ", ".join(missing))
    print("")
    print("  entry point       : " + entry)
    print("  interpreter       : " + sys.executable)
    print("  working directory : " + os.getcwd())
    print("")
    print("  Fix (Windows):")
    print("    1) double-click   setup_env.bat      (creates .venv + installs)")
    print("    2) then run       run.bat            (or webui.bat)")
    print("  Fix (Linux / macOS):")
    print("    bash setup_env.sh  &&  bash run.sh")
    print("  Or install into the current interpreter:")
    print('    "%s" -m pip install %s' % (sys.executable, " ".join(missing)))
    print("")
    print("  NOTE: on Windows `python3` is usually a 0-byte Microsoft Store")
    print("        stub that exits silently with no output. Use run.bat /")
    print("        webui.bat, or an explicit interpreter path instead.")
    print("=" * 72)
    sys.exit(2)
