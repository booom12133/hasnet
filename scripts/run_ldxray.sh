#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED=3408

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
DATA_ROOT="${DATA_ROOT:-.}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

"$TORCHRUN_BIN" \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  scripts/train.py \
  --config configs/paper/full_convnext_ldxray.yaml \
  --seed 3408 \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR"
