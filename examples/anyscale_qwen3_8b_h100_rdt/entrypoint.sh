#!/bin/bash
# Anyscale entrypoint: Qwen3-8B GRPO training on 1 worker x 8x H100-80GB
# with RDT/NIXL weight sync and local sglang/miles overlay.
#
# Head node (m5.2xlarge): driver only, no GPUs
# Layout (GPU worker):
#   Worker 0 (8x H100):
#     GPU 0-3: Training (TP=2, DP=2)
#     GPU 4-7: Rollout (4 SGLang engines, 1 GPU each)

set -ex

export PYTHONBUFFERED=16
STORAGE=/mnt/cluster_storage
CODE_DIR=${STORAGE}/local_code

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

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

# ======================== Step 0: Sync local code to shared storage ========================
# The working_dir is only available on the head node. Copy local sglang/miles
# source to shared storage so GPU workers can overlay them at runtime.
echo "=== Syncing local sglang/miles to shared storage ==="
mkdir -p ${CODE_DIR}
rsync -a --delete _bundled/sglang_python/ ${CODE_DIR}/sglang_python/
rsync -a --delete _bundled/ray_experimental/ ${CODE_DIR}/ray_experimental/
rsync -a --delete miles/ ${CODE_DIR}/miles/
cp train_async.py ${CODE_DIR}/train_async.py

# Create a wrapper script that overlays local code then runs training
cat > ${CODE_DIR}/run_training_rdt.sh << 'WRAPPER'
#!/bin/bash
set -ex
CODE_DIR=/mnt/cluster_storage/local_code

# Overlay local sglang onto Docker-installed version
SGLANG_PATH=$(python3 -c "import sglang, os; print(os.path.dirname(sglang.__file__))")
rm -rf "$SGLANG_PATH/srt" && cp -r ${CODE_DIR}/sglang_python/sglang/srt "$SGLANG_PATH/srt"
if [ -d "${CODE_DIR}/sglang_python/sglang/jit_kernel" ]; then
    rm -rf "$SGLANG_PATH/jit_kernel" && cp -r ${CODE_DIR}/sglang_python/sglang/jit_kernel "$SGLANG_PATH/jit_kernel"
fi

# Get ray install path (before overlay, so import ray still works with old experimental)
RAY_PATH=$(python3 -c "import ray, os; print(os.path.dirname(ray.__file__))")

# Patch missing constants into ray_constants.py BEFORE overlaying ray/experimental.
# The new experimental code imports these constants at module load time, so they must
# exist before the overlay. We use shell (not Python import) to avoid triggering the
# import chain after overlay.
RAY_CONSTANTS="$RAY_PATH/_private/ray_constants.py"
if ! grep -q 'NIXL_REMOTE_AGENT_CACHE_MAXSIZE' "$RAY_CONSTANTS"; then
    cat >> "$RAY_CONSTANTS" << 'PATCH'

# Patched: constants needed by ray.experimental.gpu_object_manager (Ray 3.0)
import os
NIXL_REMOTE_AGENT_CACHE_MAXSIZE = int(os.environ.get("RAY_NIXL_REMOTE_AGENT_CACHE_MAXSIZE", "1000"))
RDT_FETCH_FAIL_TIMEOUT_SECONDS = int(os.environ.get("RAY_rdt_fetch_fail_timeout_milliseconds", "60000")) / 1000
PATCH
    echo "Patched NIXL_REMOTE_AGENT_CACHE_MAXSIZE + RDT_FETCH_FAIL_TIMEOUT_SECONDS into ray_constants.py"
else
    echo "Constants already exist in ray_constants.py"
fi

# Overlay local ray/experimental onto Docker-installed version (PR #60689: send-side caching)
rm -rf "$RAY_PATH/experimental" && cp -r ${CODE_DIR}/ray_experimental "$RAY_PATH/experimental"

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
chmod +x ${CODE_DIR}/run_training_rdt.sh

# ======================== Step 1: Download model & dataset ========================

echo "=== Downloading model ==="
huggingface-cli download Qwen/Qwen3-8B --local-dir ${STORAGE}/Qwen3-8B

echo "=== Downloading dataset ==="
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k --local-dir ${STORAGE}/dapo-math-17k

# ======================== Step 2: Convert HF weights to torch_dist ========================

if [ ! -d "${STORAGE}/Qwen3-8B_torch_dist/iter_0000000" ]; then
  echo "=== Converting weights (HF -> torch_dist) on GPU worker ==="
  CONVERT_ENV_JSON='{
    "env_vars": {
      "PYTHONPATH": "/root/Megatron-LM/"
    }
  }'
  ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${CONVERT_ENV_JSON}" \
    --entrypoint-num-gpus 1 \
    -- python3 /tmp/miles/tools/convert_hf_to_torch_dist.py \
      ${MODEL_ARGS[@]} \
      --no-gradient-accumulation-fusion \
      --hf-checkpoint ${STORAGE}/Qwen3-8B \
      --save ${STORAGE}/Qwen3-8B_torch_dist
else
  echo "=== Converted weights already exist, skipping ==="
fi

# ======================== Step 3: Run training with RDT weight sync ========================

CKPT_ARGS=(
   --hf-checkpoint ${STORAGE}/Qwen3-8B
   --ref-load ${STORAGE}/Qwen3-8B_torch_dist
   --load ${STORAGE}/Qwen3-8B_torch_dist
   --save ${STORAGE}/Qwen3-8B_miles_rdt/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data ${STORAGE}/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --balance-data
   --rm-type dapo
   --reward-key score
   --num-rollout 5
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256
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
   --max-tokens-per-gpu 9216
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
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

RDT_ARGS=(
   --use-rdt-weight-sync
)

MISC_ARGS=(
   --no-gradient-accumulation-fusion
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-tensorboard
   --tensorboard-dir ${STORAGE}/tensorboard_logs
)

RUNTIME_ENV_JSON='{
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "TENSORBOARD_DIR": "/mnt/cluster_storage/tensorboard_logs",
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
    "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "NCCL_IGNORE_DISABLED_P2P": "1"
  }
}'

echo "=== Submitting training job (RDT weight sync) ==="
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   --entrypoint-num-gpus 1 \
   -- bash ${CODE_DIR}/run_training_rdt.sh \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 3 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${RDT_ARGS[@]} \
   ${MISC_ARGS[@]}
