#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}"
if [[ ! -f demo/data.json ]]; then
  .venv/bin/python src/export_demo.py
fi
.venv/bin/python demo/app.py
