"""
pack.py —— 把项目打包成一个可拷贝到任意机器的 .zip。

设计要点：
1. **不含任何机器相关信息**：排除 .venv / results / __pycache__ / .git / dist，
   因此 zip 里不会出现任何本机绝对路径；
2. 目标机器只需 Python 3.9+，解压后执行 setup_env 装依赖即可运行；
3. 只依赖标准库（zipfile），本身不需要任何第三方包。

用法：
    python pack.py                    # 生成 dist/ai-mape-k-controller-<时间戳>.zip
    python pack.py --include-results  # 连同现有实验产物一起打包
    python pack.py --output my.zip    # 指定输出文件
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_NAME = os.path.basename(HERE) or "ai-mape-k-controller"

# 需要排除的目录名（与层级无关）
EXCLUDE_DIRS = {
    ".venv", "venv", "env", "__pycache__", ".git", ".hg", ".svn",
    "dist", "build", ".idea", ".vscode", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".workbuddy",
}
# 需要排除的文件后缀
EXCLUDE_SUFFIX = (".pyc", ".pyo", ".pyd", ".so", ".log", ".tmp")
# 需要排除的具体文件（相对路径）
EXCLUDE_FILES = {".DS_Store", "Thumbs.db"}


def should_skip(dirpath: str, filename: str, include_results: bool) -> bool:
    rel_dir = os.path.relpath(dirpath, HERE).replace("\\", "/")
    parts = [p for p in rel_dir.split("/") if p not in ("", ".")]

    for p in parts:
        if p in EXCLUDE_DIRS:
            return True
        if p == "results" and not include_results:
            return True
    if filename.endswith(EXCLUDE_SUFFIX) or filename in EXCLUDE_FILES:
        return True
    # 虚拟环境内的解释器副本体积巨大且不可跨机使用
    if filename.lower().startswith("python") and parts[:1] == [".venv"]:
        return True
    return False


def build_zip(output: str, include_results: bool) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    count = 0
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=9) as zf:
        for dirpath, dirnames, filenames in os.walk(HERE):
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in EXCLUDE_DIRS
                and not (d == "results" and not include_results)
                and not (d == "dist" and os.path.abspath(dirpath) == HERE)
            )
            for fn in sorted(filenames):
                if fn == os.path.basename(output) and \
                        os.path.abspath(dirpath) == os.path.dirname(os.path.abspath(output)):
                    continue
                if should_skip(dirpath, fn, include_results):
                    continue
                full = os.path.join(dirpath, fn)
                arcname = os.path.join(PROJECT_NAME,
                                       os.path.relpath(full, HERE))
                zf.write(full, arcname.replace("\\", "/"))
                count += 1
    return output


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="打包项目为可分发的 zip（不含 .venv / 运行结果 / 缓存）")
    ap.add_argument("--output", default=None,
                    help="输出 zip 路径（默认 dist/<项目名>-<时间戳>.zip）")
    ap.add_argument("--include-results", action="store_true",
                    help="连同 results/ 下的实验产物一起打包")
    args = ap.parse_args(argv)

    if args.output:
        output = args.output if os.path.isabs(args.output) \
            else os.path.join(HERE, args.output)
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output = os.path.join(HERE, "dist", f"{PROJECT_NAME}-{stamp}.zip")

    path = build_zip(output, args.include_results)
    size = os.path.getsize(path)
    print(f"[pack] output : {path}")
    print(f"[pack] size   : {size / 1024:.1f} KiB")
    print("[pack] next   : copy the zip to the target machine, unzip, "
          "then run setup_env.bat (Windows) or bash setup_env.sh (Linux/macOS), "
          "then run.bat / run.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
