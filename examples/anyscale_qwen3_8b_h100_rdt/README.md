# Qwen3-8B GRPO on H100 - RDT Weight Sync + Local Code Overlay

Same training setup as `anyscale_qwen3_8b_h100` but with two changes:

1. **RDT/NIXL weight sync** — weights transfer from trainer to rollout engines
   via Ray Direct Transport (point-to-point RDMA) instead of NCCL broadcast.
2. **Local sglang/miles overlay** — local source is copied over the Docker-installed
   packages at runtime. No Docker rebuild needed when iterating on code.

## Cluster Layout

| Resource | Role |
|----------|------|
| Head node (m5.2xlarge) | Driver script, no GPUs |
| GPU 0-3 | Training for the default performance run (TP=2, DP=2) |
| GPU 4 | Rollout for the default performance run |
| GPU 5 | Ray job driver pinned with `--entrypoint-num-gpus 1` |

## Quick Start

```bash
# 1. Bundle local ray and sglang source (run from miles repo root)
bash examples/anyscale_qwen3_8b_h100_rdt/prepare.sh

# 2. Submit the job
anyscale job submit -f examples/anyscale_qwen3_8b_h100_rdt/job.yaml
```

The `prepare.sh` script rsyncs `../sglang/python/` into `_bundled/sglang_python/`,
which gets uploaded via `working_dir: .`. It also bundles `../ray/python/` into
`_bundled/ray_python/` when that repo is present. Set `SGLANG_DIR` or `RAY_DIR`
to override either repo location:

```bash
SGLANG_DIR=/path/to/sglang bash examples/anyscale_qwen3_8b_h100_rdt/prepare.sh
```

`prepare.sh` also checks that the local SGLang tree has the RDT APIs Miles uses:
`SchedulerActor.pull_weights`, `ParameterMapper`, `RankParallelismConfig`,
`ParallelismContext`, and the `/parallelism_config` endpoint.

## Correctness Validation

Use `job_validate.yaml` to test repeated RDT writes against the local Miles repo
and the local SGLang repo from `../sglang`.

```bash
bash examples/anyscale_qwen3_8b_h100_rdt/prepare.sh
anyscale job submit -f examples/anyscale_qwen3_8b_h100_rdt/job_validate.yaml
```

Validation mode sets `--lr 0`, uses short rollouts, and enables the existing
`--check-weight-update-equal` path. This snapshots the initial rollout tensors,
randomizes them, and verifies that the initial RDT update restores the expected
weights for each rollout topology.

The validation job runs these six-GPU-compatible cases:

| Case | Shape | What it covers |
|------|-------|----------------|
| `single` | 1 engine x TP1 | Basic RDT pull path |
| `multi_engine` | 3 engines x TP1 | One trainer source fanning out to multiple engines |
| `tp2` | 1 engine x TP2 | Per-TP-rank shard routing |

For the larger fan-out plus sharding case, run with an eight-GPU worker shape and:

```bash
VALIDATE_WEIGHT_SYNC=1 RDT_ROLLOUT_CASE=multi_engine_tp2 \
  bash examples/anyscale_qwen3_8b_h100_rdt/entrypoint.sh
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
