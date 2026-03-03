# Qwen3-8B GRPO on H100 — RDT Weight Sync + Local Code Overlay

Same training setup as `anyscale_qwen3_8b_h100` but with two changes:

1. **RDT/NIXL weight sync** — weights transfer from trainer to rollout engines
   via Ray Direct Transport (point-to-point RDMA) instead of NCCL broadcast.
2. **Local sglang/miles overlay** — local source is copied over the Docker-installed
   packages at runtime. No Docker rebuild needed when iterating on code.

## Cluster Layout

| Resource | Role |
|----------|------|
| Head node (m5.2xlarge) | Driver script, no GPUs |
| GPU 0-3 | Training (TP=2, DP=2) |
| GPU 4-7 | Rollout (SGLang engines, 1 GPU each) |

## Quick Start

```bash
# 1. Bundle local sglang source (run from miles repo root)
bash examples/anyscale_qwen3_8b_h100_rdt/prepare.sh

# 2. Submit the job
anyscale job submit -f examples/anyscale_qwen3_8b_h100_rdt/job.yaml
```

The `prepare.sh` script rsyncs `../sglang/python/` into `_bundled/sglang_python/`
which gets uploaded via `working_dir: .`. Set `SGLANG_DIR` to override the sglang
repo location:

```bash
SGLANG_DIR=/path/to/sglang bash examples/anyscale_qwen3_8b_h100_rdt/prepare.sh
```

## How the Overlay Works

The Docker image (`anyscale_qwen3_8b_h100/Dockerfile.anyscale`) has sglang and
miles pre-installed. At runtime, the wrapper script overwrites the installed
package directories with local source from shared storage:

- `sglang/srt` and `sglang/jit_kernel` are replaced in the pip-installed location
- `miles/` package directory is replaced in the pip-installed location

This gives you the compiled dependencies (sgl_kernel, flash-attn, etc.) from
Docker while using your latest Python source code.

## Differences from Base Example

| | `anyscale_qwen3_8b_h100` | This example |
|---|---|---|
| Weight sync | NCCL broadcast (`UpdateWeightFromDistributed`) | RDT/NIXL (`UpdateWeightFromRDT`) |
| Code delivery | Patches 3 files at runtime | Full sglang + miles overlay |
| Docker rebuild | Required for code changes | Not required |
| Extra env vars | — | `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`, `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` |
| Save directory | `Qwen3-8B_miles/` | `Qwen3-8B_miles_rdt/` |
