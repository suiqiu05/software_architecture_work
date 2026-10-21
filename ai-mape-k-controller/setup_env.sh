#!/usr/bin/env bash
# setup_env.sh - create .venv and install dependencies (Linux / macOS)
# Usage:  bash setup_env.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then command -v python3 >/dev/null 2>&1 && PY=python3 || true; fi
if [ -z "$PY" ]; then command -v python  >/dev/null 2>&1 && PY=python  || true; fi
if [ -z "$PY" ]; then
  echo "[ERROR] No Python interpreter found. Install Python 3.9+ first."
  exit 1
fi
echo "[1/3] base interpreter : $(command -v "$PY")"

if [ -d .venv ]; then
  echo "[2/3] .venv already exists - reusing"
else
  echo "[2/3] creating .venv ..."
  "$PY" -m venv .venv
fi

VPY=".venv/bin/python"
[ -x "$VPY" ] || VPY=".venv/Scripts/python.exe"

echo "[3/3] installing dependencies ..."
if ! "$VPY" -m pip install --disable-pip-version-check -r requirements.txt; then
  echo "[WARN] default index failed - retrying via Aliyun mirror ..."
  "$VPY" -m pip install --disable-pip-version-check \
      -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
fi

"$VPY" -c "import numpy, matplotlib; print('OK  numpy', numpy.__version__, '| matplotlib', matplotlib.__version__)"
echo
echo "Setup complete. Next:  bash run.sh"
