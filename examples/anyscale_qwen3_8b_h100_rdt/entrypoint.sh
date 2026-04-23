#!/bin/bash
# Anyscale entrypoint: Qwen3-8B GRPO training on H100 workers
# with RDT/NIXL weight sync.
#
# Head node: tiny driver, no GPUs
# Layout (1 worker node with 6 H100s, all colocated to share /tmp):
#   GPU 0-3: Training actor (TP=2 DP=2)
#   GPU 4:   Rollout (1 SGLang engine)
#   GPU 5:   Training driver (pinned via --entrypoint-num-gpus 1)

set -ex

export PYTHONBUFFERED=16
S3_BUCKET=s3://anyscale-k8s-rkn-gpu-cloud-6cc98604/miles-rdt
STORAGE=/tmp/local_storage
CODE_DIR=${STORAGE}/local_code
# Note: this staging K8s cluster does NOT have /mnt/cluster_storage. Instead we
# put model/dataset/torch_dist in S3, and the training driver — pinned to the
# GPU worker pod via `ray job submit --entrypoint-num-gpus 1` — pulls them to
# the worker pod's /tmp where actors on the same pod can read them.

# Weight sync method calibration: WEIGHT_SYNC=rdt (default) or WEIGHT_SYNC=broadcast.
# Toggles the --use-rdt-weight-sync flag and the save subdir to keep checkpoints separate.
WEIGHT_SYNC=${WEIGHT_SYNC:-rdt}
if [ "$WEIGHT_SYNC" = "broadcast" ]; then
    RDT_ARGS=()
    SAVE_SUBDIR=Qwen3-8B_miles_broadcast
elif [ "$WEIGHT_SYNC" = "rdt" ]; then
    RDT_ARGS=(--use-rdt-weight-sync)
    SAVE_SUBDIR=Qwen3-8B_miles_rdt
else
    echo "ERROR: WEIGHT_SYNC must be 'rdt' or 'broadcast', got '$WEIGHT_SYNC'" >&2
    exit 1
fi
echo "=== Weight sync method: $WEIGHT_SYNC (save: $SAVE_SUBDIR) ==="

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

if [ -n "${RDT_ROLLOUT_CASES:-}" ]; then
    IFS=',' read -ra CASE_LIST <<< "$RDT_ROLLOUT_CASES"
    for rollout_case in "${CASE_LIST[@]}"; do
        rollout_case="${rollout_case// /}"
        if [ -z "$rollout_case" ]; then
            continue
        fi
        echo "=== Running RDT rollout validation case: $rollout_case ==="
        RDT_ROLLOUT_CASE="$rollout_case" RDT_ROLLOUT_CASES= bash "$SCRIPT_DIR/entrypoint.sh"
    done
    exit 0
fi

# Qwen3-8B model architecture args (from scripts/models/qwen3-8B.sh)
MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 4096
   --ffn-hidden-size 12288
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 1000000
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
   --untie-embeddings-and-output-weights
)

# RDT_ROLLOUT_CASE selects the rollout topology to validate:
#   single:          1 engine x TP1, default 6-GPU job shape.
#   multi_engine:    3 engines x TP1, tests one source serving multiple engines.
#   tp2:             1 engine x TP2, tests sharded rollout ranks.
#   multi_engine_tp2: 2 engines x TP2, needs at least 7 GPUs with the driver GPU.
RDT_ROLLOUT_CASE=${RDT_ROLLOUT_CASE:-single}
case "$RDT_ROLLOUT_CASE" in
    single)
        ACTOR_NUM_GPUS_PER_NODE=4
        ROLLOUT_NUM_GPUS=1
        ROLLOUT_NUM_GPUS_PER_ENGINE=1
        ;;
    multi_engine)
        ACTOR_NUM_GPUS_PER_NODE=2
        ROLLOUT_NUM_GPUS=3
        ROLLOUT_NUM_GPUS_PER_ENGINE=1
        ;;
    tp2)
        ACTOR_NUM_GPUS_PER_NODE=2
        ROLLOUT_NUM_GPUS=2
        ROLLOUT_NUM_GPUS_PER_ENGINE=2
        ;;
    multi_engine_tp2)
        ACTOR_NUM_GPUS_PER_NODE=2
        ROLLOUT_NUM_GPUS=4
        ROLLOUT_NUM_GPUS_PER_ENGINE=2
        ;;
    *)
        echo "ERROR: unknown RDT_ROLLOUT_CASE '$RDT_ROLLOUT_CASE'" >&2
        exit 1
        ;;
esac

VALIDATE_WEIGHT_SYNC=${VALIDATE_WEIGHT_SYNC:-0}
CHECK_ARGS=()
if [ "$VALIDATE_WEIGHT_SYNC" = "1" ]; then
    NUM_ROLLOUT=${NUM_ROLLOUT:-3}
    ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-4}
    N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
    ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-128}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
    MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-512}
    LR=${LR:-0}
    SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
    CHECK_ARGS=(--ci-test --check-weight-update-equal --update-weights-interval 1)
else
    NUM_ROLLOUT=${NUM_ROLLOUT:-5}
    ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
    N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
    ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-8192}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
    MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-4096}
    LR=${LR:-1e-6}
    SAVE_INTERVAL=${SAVE_INTERVAL:-20}
fi
echo "=== Rollout case: $RDT_ROLLOUT_CASE (actor_gpus=$ACTOR_NUM_GPUS_PER_NODE rollout_gpus=$ROLLOUT_NUM_GPUS per_engine=$ROLLOUT_NUM_GPUS_PER_ENGINE validate=$VALIDATE_WEIGHT_SYNC) ==="

# ======================== Step 0: Sync code to GCS shared storage ========================
# K8s pods don't share a filesystem. Upload code to GCS so workers can pull it.
echo "=== Syncing code to GCS ==="
mkdir -p ${CODE_DIR}

# Use the sibling local sglang repo bundled by prepare.sh.
if [ ! -d "_bundled/sglang_python/sglang" ]; then
    echo "ERROR: missing _bundled/sglang_python/sglang."
    echo "Run: bash examples/anyscale_qwen3_8b_h100_rdt/prepare.sh"
    exit 1
fi
rm -rf ${STORAGE}/sglang_python
rsync -a --delete _bundled/sglang_python/ ${STORAGE}/sglang_python/

# Patch sglang dumper.py to add DumperConfig stub (Docker's miles imports it)
DUMPER_PY="${STORAGE}/sglang_python/sglang/srt/debug_utils/dumper.py"
if [ -f "$DUMPER_PY" ] && ! grep -q "DumperConfig" "$DUMPER_PY"; then
    printf '\n# Backward compat stub for older miles Docker images\nDumperConfig = None\n_get_rank = lambda: 0\ndumper = None\n' >> "$DUMPER_PY"
    echo "=== Patched sglang dumper.py with DumperConfig stub ==="
fi
# Prepare working_code dir for Ray working_dir distribution to workers.
# Ray inserts working_dir at sys.path[0] on every worker, so "import miles"
# finds OUR miles (with try/except in dumper_utils.py) before Docker's editable install.
rm -rf ${STORAGE}/working_code && mkdir -p ${STORAGE}/working_code
cp -r miles ${STORAGE}/working_code/

# Upload sglang, miles, and train_async.py to GCS
aws s3 sync --delete \
    ${STORAGE}/sglang_python/ ${S3_BUCKET}/code/sglang_python/
aws s3 sync --delete \
    miles/ ${S3_BUCKET}/code/miles/
aws s3 cp train_async.py ${S3_BUCKET}/code/train_async.py

# Create wrapper script and upload to GCS
cat > ${STORAGE}/run_training_rdt.sh << 'WRAPPER'
#!/bin/bash
set -ex
S3_BUCKET=s3://anyscale-k8s-rkn-gpu-cloud-6cc98604/miles-rdt
CODE_DIR=/tmp/local_code
mkdir -p ${CODE_DIR}

# Pull code from S3
aws s3 sync ${S3_BUCKET}/code/sglang_python/ ${CODE_DIR}/sglang_python/
aws s3 sync ${S3_BUCKET}/code/miles/ ${CODE_DIR}/miles/
aws s3 cp ${S3_BUCKET}/code/train_async.py ${CODE_DIR}/train_async.py

# Pull model/dataset/torch_dist from S3 to this pod's /tmp. Driver runs here
# via --entrypoint-num-gpus 1 so the actors scheduled on this same worker pod
# share the same filesystem.
aws s3 sync ${S3_BUCKET}/Qwen3-8B/ /tmp/Qwen3-8B/
aws s3 sync ${S3_BUCKET}/Qwen3-8B_torch_dist/ /tmp/Qwen3-8B_torch_dist/
aws s3 sync ${S3_BUCKET}/dapo-math-17k/ /tmp/dapo-math-17k/

# Overlay sglang from PR branch onto Docker-installed version
SGLANG_PATH=$(python3 -c "import sglang, os; print(os.path.dirname(sglang.__file__))")
rm -rf "$SGLANG_PATH/srt" && cp -r ${CODE_DIR}/sglang_python/sglang/srt "$SGLANG_PATH/srt"
if [ -d "${CODE_DIR}/sglang_python/sglang/jit_kernel" ]; then
    rm -rf "$SGLANG_PATH/jit_kernel" && cp -r ${CODE_DIR}/sglang_python/sglang/jit_kernel "$SGLANG_PATH/jit_kernel"
fi
cp -f ${CODE_DIR}/sglang_python/sglang/utils.py "$SGLANG_PATH/utils.py"

# Upgrade flashinfer to match local sglang's requirements
pip install --no-cache-dir -q flashinfer_python==0.6.3 flashinfer_cubin==0.6.3

# Install NIXL for Ray RDT tensor transport
pip install --no-cache-dir -q nixl

# Overlay local miles onto Docker-installed version
MILES_PATH=$(python3 -c "import miles, os; print(os.path.dirname(miles.__file__))")
rm -rf "$MILES_PATH" && cp -r ${CODE_DIR}/miles/ "$MILES_PATH/"

echo "=== Local sglang/miles overlaid ==="
exec python3 ${CODE_DIR}/train_async.py "$@"
WRAPPER
chmod +x ${STORAGE}/run_training_rdt.sh
aws s3 cp ${STORAGE}/run_training_rdt.sh ${S3_BUCKET}/code/run_training_rdt.sh

# Create setup overlay script and upload to S3.
# setup_commands in RUNTIME_ENV_JSON runs this on ALL Ray worker pods before
# any actor is scheduled, ensuring sglang/miles overlay happens everywhere
# (not just the training driver pod).
# IMPORTANT: Use hardcoded Docker editable-install paths, NOT dynamic import.
# When called from setup_commands with working_dir in sys.path, "import miles"
# would resolve to the Ray packages path, not Docker's /tmp/miles/miles/.
# Actors without working_dir use Docker's editable installs, so we must patch
# those paths directly.
cat > ${STORAGE}/setup_overlay.sh << 'SETUP_EOF'
#!/bin/bash
set -ex
S3_BUCKET=s3://anyscale-k8s-rkn-gpu-cloud-6cc98604/miles-rdt
CODE_DIR=/tmp/local_code
mkdir -p ${CODE_DIR}

# Overlay sglang from PR branch onto Docker-installed version.
# Docker installs sglang as editable at /home/ray/sglang/python/sglang.
aws s3 sync ${S3_BUCKET}/code/sglang_python/ ${CODE_DIR}/sglang_python/
SGLANG_DOCKER_PATH=/home/ray/sglang/python/sglang
rm -rf "$SGLANG_DOCKER_PATH/srt" && cp -r ${CODE_DIR}/sglang_python/sglang/srt "$SGLANG_DOCKER_PATH/srt"
if [ -d "${CODE_DIR}/sglang_python/sglang/jit_kernel" ]; then
    rm -rf "$SGLANG_DOCKER_PATH/jit_kernel" && cp -r ${CODE_DIR}/sglang_python/sglang/jit_kernel "$SGLANG_DOCKER_PATH/jit_kernel"
fi
cp -f ${CODE_DIR}/sglang_python/sglang/utils.py "$SGLANG_DOCKER_PATH/utils.py"

# Overlay local miles onto Docker-installed version.
# Docker installs miles as editable at /tmp/miles/miles.
aws s3 sync ${S3_BUCKET}/code/miles/ ${CODE_DIR}/miles/
MILES_DOCKER_PATH=/tmp/miles/miles
rm -rf "$MILES_DOCKER_PATH" && cp -r ${CODE_DIR}/miles/ "$MILES_DOCKER_PATH/"

# Install required packages
pip install --no-cache-dir -q flashinfer_python==0.6.3 flashinfer_cubin==0.6.3
pip install --no-cache-dir -q nixl

echo "=== sglang/miles overlay complete on $(hostname) ==="
SETUP_EOF
chmod +x ${STORAGE}/setup_overlay.sh
aws s3 cp ${STORAGE}/setup_overlay.sh ${S3_BUCKET}/code/setup_overlay.sh

# ======================== Step 1: Download model & dataset to GCS ========================

echo "=== Checking model in S3 ==="
if ! aws s3 ls ${S3_BUCKET}/Qwen3-8B/config.json 2>/dev/null; then
  echo "=== Downloading model from HuggingFace ==="
  huggingface-cli download Qwen/Qwen3-8B --local-dir /tmp/Qwen3-8B
  echo "=== Uploading model to S3 ==="
  aws s3 sync /tmp/Qwen3-8B/ ${S3_BUCKET}/Qwen3-8B/
else
  echo "=== Model already in S3, skipping ==="
fi

echo "=== Checking dataset in S3 ==="
if ! aws s3 ls ${S3_BUCKET}/dapo-math-17k/dapo-math-17k.jsonl 2>/dev/null; then
  echo "=== Downloading dataset from HuggingFace ==="
  huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /tmp/dapo-math-17k
  echo "=== Uploading dataset to S3 ==="
  aws s3 sync /tmp/dapo-math-17k/ ${S3_BUCKET}/dapo-math-17k/
else
  echo "=== Dataset already in S3, skipping ==="
fi

# ======================== Step 2: Convert HF weights to torch_dist ========================

if ! aws s3 ls ${S3_BUCKET}/Qwen3-8B_torch_dist/latest_checkpointed_iteration.txt 2>/dev/null; then
  echo "=== Converting weights (HF -> torch_dist) on GPU worker ==="

  # Write conversion script (avoids bash -c array expansion issues)
  cat > ${STORAGE}/convert_weights.sh << CONVERT_EOF
#!/bin/bash
set -ex
aws s3 sync ${S3_BUCKET}/Qwen3-8B/ /tmp/Qwen3-8B/
python3 /tmp/miles/tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --no-gradient-accumulation-fusion \
  --hf-checkpoint /tmp/Qwen3-8B \
  --save /tmp/Qwen3-8B_torch_dist
aws s3 sync /tmp/Qwen3-8B_torch_dist/ ${S3_BUCKET}/Qwen3-8B_torch_dist/
CONVERT_EOF
  chmod +x ${STORAGE}/convert_weights.sh
  aws s3 cp ${STORAGE}/convert_weights.sh ${S3_BUCKET}/code/convert_weights.sh

  CONVERT_ENV_JSON='{
    "env_vars": {
      "PYTHONPATH": "/home/ray/Megatron-LM/"
    }
  }'
  ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${CONVERT_ENV_JSON}" \
    --entrypoint-num-gpus 1 \
    -- bash -c 'aws s3 cp s3://anyscale-k8s-rkn-gpu-cloud-6cc98604/miles-rdt/code/convert_weights.sh /tmp/convert_weights.sh && bash /tmp/convert_weights.sh'
else
  echo "=== Converted weights already in S3, skipping ==="
fi

# ======================== Step 3: Run training with RDT weight sync ========================

CKPT_ARGS=(
   --hf-checkpoint /tmp/Qwen3-8B
   --ref-load /tmp/Qwen3-8B_torch_dist
   --load /tmp/Qwen3-8B_torch_dist
   --save /tmp/${SAVE_SUBDIR}/
   --save-interval ${SAVE_INTERVAL}
)

ROLLOUT_ARGS=(
   --prompt-data /tmp/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type dapo
   --reward-key score
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}
   --rollout-temperature 1
   --global-batch-size ${GLOBAL_BATCH_SIZE}
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR}
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --no-gradient-accumulation-fusion
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-tensorboard
   --tensorboard-dir /tmp/tensorboard_logs
)

RUNTIME_ENV_JSON='{
  "working_dir": "/tmp/local_storage/working_code",
  "env_vars": {
    "PYTHONPATH": "/home/ray/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "TENSORBOARD_DIR": "/tmp/tensorboard_logs",
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
    "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "NCCL_IGNORE_DISABLED_P2P": "1",
    "DEPRECATED_MEGATRON_COMPATIBLE": "1"
  },
  "setup_commands": [
    "aws s3 cp s3://anyscale-k8s-rkn-gpu-cloud-6cc98604/miles-rdt/code/setup_overlay.sh /tmp/setup_overlay.sh && bash /tmp/setup_overlay.sh"
  ]
}'

echo "=== Submitting training job (RDT weight sync) ==="
# --entrypoint-num-gpus 1: pin the driver onto the GPU worker pod so it
# shares /tmp with the actors it spawns on the same pod. The driver pulls
# run_training_rdt.sh from S3 since /tmp on the head pod is not visible here.
# Args are inlined into the bash -c command (rather than positional via "$@")
# because Ray's outer shell would otherwise expand $@ in its own (empty) scope.
TRAINING_ARGS="--actor-num-nodes 1 --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE} --rollout-num-gpus ${ROLLOUT_NUM_GPUS} \
   ${MODEL_ARGS[*]} ${CKPT_ARGS[*]} ${ROLLOUT_ARGS[*]} ${OPTIMIZER_ARGS[*]} \
   ${GRPO_ARGS[*]} ${PERF_ARGS[*]} ${SGLANG_ARGS[*]} ${RDT_ARGS[*]} ${MISC_ARGS[*]} ${CHECK_ARGS[*]}"
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --entrypoint-num-gpus 1 \
   -- bash -c "aws s3 cp ${S3_BUCKET}/code/run_training_rdt.sh /tmp/run_training_rdt.sh && bash /tmp/run_training_rdt.sh ${TRAINING_ARGS}"
