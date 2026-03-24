# Qwen3-8B GRPO on H100 — RDT Weight Sync + Local Code Overlay

Same training setup as `anyscale_qwen3_8b_h100` but with two changes:

1. **RDT/NIXL weight sync** — weights transfer from trainer to rollout engines
   via Ray Direct Transport (point-to-point RDMA) instead of NCCL broadcast.
2. **Local miles overlay** — local miles source is copied over the Docker-installed
   package at runtime. No Docker rebuild needed when iterating on code.

## Cluster Layout

| Resource | Role |
|----------|------|
| Head node (m5.2xlarge) | Driver script, no GPUs |
| GPU 0-3 | Training (TP=2, DP=2) |
| GPU 4-6 | Rollout (3 SGLang engines, 1 GPU each) |

## Quick Start

```bash
anyscale job submit -f examples/anyscale_qwen3_8b_h100_rdt/job.yaml
```

## How the Overlay Works

The Docker image (`anyscale_qwen3_8b_h100/Dockerfile.anyscale`) has sglang and
miles pre-installed. The entrypoint rsyncs local `miles/` to shared storage, then
the wrapper script on the GPU worker overwrites the pip-installed miles package
with the local source.

This gives you the compiled dependencies (sgl_kernel, flash-attn, etc.) from
Docker while using your latest miles source code.

## Differences from Base Example

| | `anyscale_qwen3_8b_h100` | This example |
|---|---|---|
| Weight sync | NCCL broadcast (`UpdateWeightFromDistributed`) | RDT/NIXL (`UpdateWeightFromRDT`) |
| Code delivery | Patches 3 files at runtime | Local miles overlay via shared storage |
| Docker rebuild | Required for code changes | Not required |
| Extra env vars | — | `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`, `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` |
| Save directory | `Qwen3-8B_miles/` | `Qwen3-8B_miles_rdt/` |
