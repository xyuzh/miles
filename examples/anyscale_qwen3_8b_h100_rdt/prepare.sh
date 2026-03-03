#!/bin/bash
# Bundle local sglang source into _bundled/ so it gets uploaded via working_dir.
# Run from the miles repo root before `anyscale job submit`.
#
# Usage:
#   bash examples/anyscale_qwen3_8b_h100_rdt/prepare.sh
#   anyscale job submit -f examples/anyscale_qwen3_8b_h100_rdt/job.yaml

set -ex

SGLANG_DIR="${SGLANG_DIR:-../sglang}"
RAY_DIR="${RAY_DIR:-../ray}"

if [ ! -d "${SGLANG_DIR}/python/sglang" ]; then
    echo "ERROR: sglang not found at ${SGLANG_DIR}/python/sglang"
    echo "Set SGLANG_DIR to the sglang repo root, e.g.: SGLANG_DIR=../sglang bash $0"
    exit 1
fi

if [ ! -d "${RAY_DIR}/python/ray/experimental" ]; then
    echo "ERROR: ray not found at ${RAY_DIR}/python/ray/experimental"
    echo "Set RAY_DIR to the ray repo root, e.g.: RAY_DIR=../ray bash $0"
    exit 1
fi

mkdir -p _bundled/sglang_python
rsync -a --delete "${SGLANG_DIR}/python/" _bundled/sglang_python/

mkdir -p _bundled/ray_experimental
rsync -a --delete "${RAY_DIR}/python/ray/experimental/" _bundled/ray_experimental/

echo "=== Bundled sglang from ${SGLANG_DIR}/python/ ==="
echo "=== Bundled ray/experimental from ${RAY_DIR}/python/ray/experimental/ ==="
echo "Now run: anyscale job submit -f examples/anyscale_qwen3_8b_h100_rdt/job.yaml"
