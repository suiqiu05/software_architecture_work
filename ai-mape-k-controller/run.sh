#!/usr/bin/env bash
# run.sh - run the experiment (Linux / macOS)
# Usage:  bash run.sh                # default scenario
#         bash run.sh --profile burst --steps 600
set -euo pipefail
cd "$(dirname "$0")"

if [ -x .venv/bin/python ]; then
  VPY=".venv/bin/python"
elif [ -x .venv/Scripts/python.exe ]; then
  VPY=".venv/Scripts/python.exe"
else
  VPY="${PYTHON:-}"
  if [ -z "$VPY" ]; then command -v python3 >/dev/null 2>&1 && VPY=python3 || true; fi
  if [ -z "$VPY" ]; then command -v python  >/dev/null 2>&1 && VPY=python  || true; fi
  if [ -z "$VPY" ]; then
    echo "[ERROR] No usable environment. Run:  bash setup_env.sh"
    exit 1
  fi
fi

echo "Using interpreter: $VPY"
exec "$VPY" main_exp.py "$@"
