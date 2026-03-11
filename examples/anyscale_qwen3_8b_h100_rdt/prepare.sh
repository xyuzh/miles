#!/bin/bash
# Bundle local sglang source into _bundled/ so it gets uploaded via working_dir.
# Run from the miles repo root before `anyscale job submit`.
#
# Usage:
#   bash examples/anyscale_qwen3_8b_h100_rdt/prepare.sh
#   anyscale job submit -f examples/anyscale_qwen3_8b_h100_rdt/job.yaml

set -ex

SGLANG_DIR="${SGLANG_DIR:-../sglang}"

if [ ! -d "${SGLANG_DIR}/python/sglang" ]; then
    echo "ERROR: sglang not found at ${SGLANG_DIR}/python/sglang"
    echo "Set SGLANG_DIR to the sglang repo root, e.g.: SGLANG_DIR=../sglang bash $0"
    exit 1
fi

mkdir -p _bundled/sglang_python
rsync -a --delete "${SGLANG_DIR}/python/" _bundled/sglang_python/

echo "=== Bundled sglang from ${SGLANG_DIR}/python/ ==="
echo "Now run: anyscale job submit -f examples/anyscale_qwen3_8b_h100_rdt/job.yaml"
