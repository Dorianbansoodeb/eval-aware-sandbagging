#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${PWD}/.cache/huggingface"
export PYTHONPATH="${PWD}"
./.venv/bin/python src/run.py --stage all
