#!/bin/bash
# Bundle local ray and sglang source into _bundled/ so they get uploaded via working_dir.
# Run from the miles repo root before `anyscale job submit`.
#
# Usage:
#   bash examples/anyscale_qwen3_8b_h100_rdt/prepare.sh
#   ANYSCALE_HOST=https://console.anyscale-staging.com \
#     anyscale job submit -f examples/anyscale_qwen3_8b_h100_rdt/job.yaml

set -ex

RAY_DIR="${RAY_DIR:-../ray}"
SGLANG_DIR="${SGLANG_DIR:-../sglang}"

bundle_ray=1
if [ ! -d "${RAY_DIR}/python/ray" ]; then
    echo "WARN: ray not found at ${RAY_DIR}/python/ray; skipping ray bundle"
    bundle_ray=0
fi

if [ ! -d "${SGLANG_DIR}/python/sglang" ]; then
    echo "ERROR: sglang not found at ${SGLANG_DIR}/python/sglang"
    echo "Set SGLANG_DIR to the sglang repo root, e.g.: SGLANG_DIR=../sglang bash $0"
    exit 1
fi

missing_sglang_api=0
if [ ! -f "${SGLANG_DIR}/python/sglang/srt/model_loader/parameter_mapper.py" ]; then
    echo "ERROR: sglang is missing srt/model_loader/parameter_mapper.py required by UpdateWeightFromRDT"
    missing_sglang_api=1
fi
if ! grep -q "RankParallelismConfig" "${SGLANG_DIR}/python/sglang/srt/distributed/parallel_state.py"; then
    echo "ERROR: sglang is missing RankParallelismConfig required by UpdateWeightFromRDT"
    missing_sglang_api=1
fi
if ! grep -q "ParallelismContext" "${SGLANG_DIR}/python/sglang/srt/distributed/parallel_state.py"; then
    echo "ERROR: sglang is missing ParallelismContext required by UpdateWeightFromRDT"
    missing_sglang_api=1
fi
if ! grep -R -q "parallelism_config" "${SGLANG_DIR}/python/sglang/srt/entrypoints" "${SGLANG_DIR}/python/sglang/srt/managers"; then
    echo "ERROR: sglang is missing the /parallelism_config endpoint required by Miles"
    missing_sglang_api=1
fi
if ! grep -q "def pull_weights" "${SGLANG_DIR}/python/sglang/srt/ray/scheduler_actor.py"; then
    echo "ERROR: sglang SchedulerActor is missing pull_weights required by RDT"
    missing_sglang_api=1
fi
if [ "$missing_sglang_api" -ne 0 ]; then
    exit 1
fi

if [ "$bundle_ray" -eq 1 ]; then
    mkdir -p _bundled/ray_python
    rsync -a --delete "${RAY_DIR}/python/" _bundled/ray_python/
    echo "=== Bundled ray from ${RAY_DIR}/python/ ==="
fi

mkdir -p _bundled/sglang_python
rsync -a --delete "${SGLANG_DIR}/python/" _bundled/sglang_python/

echo "=== Bundled sglang from ${SGLANG_DIR}/python/ ==="
echo "Now run: ANYSCALE_HOST=https://console.anyscale-staging.com anyscale job submit -f examples/anyscale_qwen3_8b_h100_rdt/job.yaml"
