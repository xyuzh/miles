# Weight Sync Pipeline: Non-Colocated Case

> How trainer weights are synced to the inference engine in Miles, with GPU memory deep dive.

## Table of Contents

- [1. Architecture Overview](#1-architecture-overview)
- [2. Entry Point: Training Loop](#2-entry-point-training-loop)
- [3. Weight Updater Selection](#3-weight-updater-selection)
- [4. Actor `update_weights()` Method](#4-actor-update_weights-method)
- [5. NCCL Group Creation](#5-nccl-group-creation)
- [6. Core Weight Broadcast (Megatron)](#6-core-weight-broadcast-megatron)
- [7. Core Weight Broadcast (FSDP)](#7-core-weight-broadcast-fsdp)
- [8. Engine-Side Reception (SGLang)](#8-engine-side-reception-sglang)
- [9. GPU Memory Deep Dive: TP All-Gather](#9-gpu-memory-deep-dive-tp-all-gather)
- [10. GPU Memory Deep Dive: EP All-Gather (MoE)](#10-gpu-memory-deep-dive-ep-all-gather-moe)
- [11. GPU Memory Deep Dive: Engine-Side](#11-gpu-memory-deep-dive-engine-side)
- [12. Async Overlap Analysis](#12-async-overlap-analysis)
- [13. Key Files Reference](#13-key-files-reference)

---

## 1. Architecture Overview

In **non-colocated mode** (default), training actors and inference engines run on **separate GPUs/nodes**. Weight sync uses **NCCL broadcast** from trainer rank 0 to all engine GPUs.

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                     Non-Colocated Architecture                      │
 │                                                                     │
 │  Training GPUs (TP/PP/DP)           Inference GPUs (SGLang engines) │
 │  ┌─────┐ ┌─────┐ ┌─────┐           ┌─────┐ ┌─────┐ ┌─────┐       │
 │  │ GPU │ │ GPU │ │ GPU │           │ GPU │ │ GPU │ │ GPU │       │
 │  │  0  │ │  1  │ │  2  │           │  3  │ │  4  │ │  5  │       │
 │  │Actor│ │Actor│ │Actor│           │Eng 0│ │Eng 0│ │Eng 1│       │
 │  │TP=0 │ │TP=1 │ │TP=2 │           │TP=0 │ │TP=1 │ │TP=0 │       │
 │  └──┬──┘ └──┬──┘ └──┬──┘           └──┬──┘ └──┬──┘ └──┬──┘       │
 │     │       │       │                  │       │       │           │
 │     └───────┼───────┘                  └───────┼───────┘           │
 │             │                                  │                   │
 │        TP All-Gather                    NCCL Broadcast             │
 │        (rank 0 gathers)              (rank 0 → engines)           │
 │             │                                  │                   │
 │             └──────────────────────────────────┘                   │
 │                        Weight Sync                                 │
 └─────────────────────────────────────────────────────────────────────┘
```

**Colocated vs non-colocated:**

| Aspect | Colocated | Non-Colocated |
|--------|-----------|---------------|
| GPU sharing | Same GPUs for train + inference | Separate GPU pools |
| Transport | Gloo (CPU) + Ray IPC | NCCL (GPU-to-GPU) |
| Updater class | `UpdateWeightFromTensor` | `UpdateWeightFromDistributed` |
| Serialization | `FlattenedTensorBucket` + `MultiprocessingSerializer` | Direct tensor broadcast |
| Latency | Higher (serialization overhead) | Lower (direct GPU broadcast) |

---

## 2. Entry Point: Training Loop

**File:** `train.py:89`

After each training iteration, the training loop calls `update_weights()`:

```python
# train.py — main training loop
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

    # ... training happens here ...
    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

    # ... save checkpoint ...

    offload_train()                        # free training GPU memory if needed
    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    actor_model.update_weights()           # ← WEIGHT SYNC HAPPENS HERE

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())
```

The initial weight sync at line 27 ensures SGLang has the loaded weights from the checkpoint before the first rollout.

---

## 3. Weight Updater Selection

Both FSDP and Megatron actors select the updater at initialization time based on `args.colocate`:

**FSDP** (`backends/fsdp_utils/actor.py:141-144`):

```python
self.weight_updater = (
    UpdateWeightFromTensor(self.args, self.model)   # colocated → Gloo IPC
    if self.args.colocate
    else UpdateWeightFromDistributed(self.args, self.model)  # non-colocated → NCCL
)
```

**Megatron** (`backends/megatron_utils/actor.py:134-141`):

```python
update_weight_cls = UpdateWeightFromTensor if self.args.colocate else UpdateWeightFromDistributed
self.weight_updater = update_weight_cls(
    self.args, self.model,
    weights_getter=lambda: self.weights_backuper.get("actor"),
    model_name=...,
    quantization_config=...,
)
```

---

## 4. Actor `update_weights()` Method

Both backends follow the same structure:

**Megatron** (`backends/megatron_utils/actor.py:464-498`):

```python
def update_weights(self) -> None:
    # 1. Recover failed engines (fault tolerance)
    if self.args.use_fault_tolerance:
        if dist.get_rank() == 0:
            ray.get(self.rollout_manager.recover_rollout_engines.remote())
        dist.barrier(group=get_gloo_group())

    # 2. Get engine handles from RolloutManager
    rollout_engines, rollout_engine_lock, num_new_engines = ray.get(
        self.rollout_manager.get_rollout_engines_and_lock.remote()
    )

    # 3. Connect to engines (creates NCCL groups if first time or after recovery)
    if num_new_engines > 0:
        self.weight_updater.connect_rollout_engines(rollout_engines, rollout_engine_lock)

    # 4. Perform weight sync
    self.weight_updater.update_weights()

    # 5. Validate consistency (CI test mode)
    if self.args.ci_test and len(rollout_engines) > 0:
        engine = random.choice(rollout_engines)
        engine_version = ray.get(engine.get_weight_version.remote())
        if str(engine_version) != str(self.weight_updater.weight_version):
            raise RuntimeError("Weight version mismatch!")
```

**FSDP** (`backends/fsdp_utils/actor.py:541-569`): Same pattern, without fault tolerance and offload handling.

---

## 5. NCCL Group Creation

### FSDP (`fsdp_utils/update_weight_utils.py:180-220`)

Only **rank 0** participates. Creates a single NCCL group `"miles"`:

```python
world_size = self.args.rollout_num_gpus + 1   # +1 for trainer rank 0

self._model_update_groups = init_process_group(
    backend="nccl",
    init_method=f"tcp://{master_address}:{master_port}",
    world_size=world_size,
    rank=0,
    group_name="miles",
)

# Each engine joins the same group via Ray RPC
for i, engine in enumerate(rollout_engines):
    engine.init_weights_update_group.remote(
        master_address, master_port,
        rank_offset=i * args.rollout_num_gpus_per_engine + 1,
        world_size=world_size, group_name="miles", backend="nccl",
    )
```

### Megatron (`megatron_utils/update_weight/update_weight_from_distributed.py:45-71`)

Creates **one NCCL group per PP stage**, only on the PP source rank (DP=0, TP=0):

```python
group_name = f"miles-pp_{pp_rank}"
```

```
 NCCL Group Layout (example: PP=2, 2 engines each with TP=4)
 ═══════════════════════════════════════════════════════════════

 Group "miles-pp_0":
 ┌──────────┬─────────────────────────────────────────────┐
 │  Rank 0  │  Trainer PP0 (DP=0, TP=0)                  │
 │  Rank 1  │  Engine 0, TP rank 0  (rank_offset=1)      │
 │  Rank 2  │  Engine 0, TP rank 1                       │
 │  Rank 3  │  Engine 0, TP rank 2                       │
 │  Rank 4  │  Engine 0, TP rank 3                       │
 │  Rank 5  │  Engine 1, TP rank 0  (rank_offset=5)      │
 │  ...     │  ...                                       │
 └──────────┴─────────────────────────────────────────────┘

 Group "miles-pp_1":
 ┌──────────┬─────────────────────────────────────────────┐
 │  Rank 0  │  Trainer PP1 (DP=0, TP=0)                  │
 │  Rank 1  │  Engine 0, TP rank 0                       │
 │  ...     │  ...                                       │
 └──────────┴─────────────────────────────────────────────┘

 world_size = 1 + num_engines × rollout_num_gpus_per_engine
```

**GPU memory**: NCCL communicator state is ~tens of MB per group, persistent for the group's lifetime.

---

## 6. Core Weight Broadcast (Megatron)

**File:** `backends/megatron_utils/update_weight/update_weight_from_distributed.py:73-133`

### Overall Flow

```
 Pause → Flush → [Non-expert params] → Barrier → [Expert params] → Continue → Post-process
```

### Step-by-step

#### a. Pause inference (line 80-82)

```python
ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
```

Stops inference and frees KV cache memory on engine GPUs.

#### b. Non-expert parameters (line 92-106)

For each parameter:

```python
for name, param in named_params_and_buffers(self.args, self.model):
    if ".experts." in name:
        continue
    # 1. TP all-gather → full param (synchronous)
    param = all_gather_param(name, param)

    # 2. Convert Megatron → HuggingFace format
    converted_named_tensors += convert_to_hf(args, model_name, name, param, ...)

    # 3. When buffer exceeds threshold → flush (NCCL broadcast)
    if buffer_size + param_size > update_weight_buffer_size:
        _update_bucket_weights_from_distributed(converted_named_tensors)
        buffer_size = 0
```

#### c. Bucket broadcast with locking (line 221-242)

```python
def _update_bucket_weights_from_distributed(self, converted_named_tensors):
    # Lock prevents NCCL deadlock across PP stages
    while not ray.get(self.rollout_engine_lock.acquire.remote()):
        time.sleep(0.1)

    # Two-phase protocol:
    # Phase 1 — Metadata via Ray RPC (small, fast)
    refs = [engine.update_weights_from_distributed.remote(
        names=[...], dtypes=[...], shapes=[...], group_name=group_name,
    ) for engine in rollout_engines]

    # Phase 2 — Tensor data via NCCL (bulk GPU-to-GPU)
    handles = []
    for _, param in converted_named_tensors:
        handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    ray.get(refs)
    converted_named_tensors.clear()
    ray.get(self.rollout_engine_lock.release.remote())
```

#### d. Expert parameters (line 109-119)

Expert params require an additional **EP (Expert Parallel) all-gather** before broadcast. The buffer threshold accounts for EP multiplication:

```python
if (buffer_size + param_size) * ep_world_size > update_weight_buffer_size:
    flush()
```

#### e. Resume inference (line 121-133)

```python
ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
```

For quantized models (int4/fp4/mxfp8), a post-processing step re-quantizes on the engine GPUs.

---

## 7. Core Weight Broadcast (FSDP)

**File:** `backends/fsdp_utils/update_weight_utils.py:46-76, 222-258`

### DTensor Redistribution

FSDP2 uses `DTensor` with sharded placements. The all-gather is implicit and **async**:

```python
def update_weights(self) -> None:
    for name, param in self.model.state_dict().items():
        param = param.cuda()
        if isinstance(param, DTensor):
            param = param.redistribute(              # async: starts all-gather
                placements=[Replicate()] * param.device_mesh.ndim,
                async_op=True,
            ).to_local()                             # returns AsyncCollectiveTensor
        bucket.append((name, param))                 # param is a "future"

        if bucket_size >= update_weight_buffer_size:
            self.wait_and_update_bucket_weights(bucket)  # .wait() materializes

def wait_and_update_bucket_weights(self, bucket):
    bucket = [(name, param.wait()) if hasattr(param, "wait") else (name, param)
              for name, param in bucket]
    self.update_bucket_weights(bucket)
```

### NCCL Broadcast (per bucket)

```python
def update_bucket_weights(self, named_tensors, weight_version=None):
    # Send metadata to engines
    refs = [engine.update_weights_from_distributed.remote(
        names=[...], dtypes=[...], shapes=[...], group_name="miles",
    ) for engine in self.rollout_engines]

    # Broadcast tensors
    handles = []
    for _name, param in named_tensors:
        torch.cuda.empty_cache()                   # free fragmented memory
        param_data = param.data.contiguous()        # NCCL requires contiguous
        handles.append(dist.broadcast(param_data, 0, group=group, async_op=True))

    for handle in handles:
        handle.wait()
    ray.get(refs)
```

---

## 8. Engine-Side Reception (SGLang)

### Request Flow Through SGLang

```
 Miles Trainer (Ray RPC)
       │
       ▼
 ┌─────────────────────────────────────────────────────┐
 │  SGLangEngine (Ray Actor)                           │
 │  sglang_engine.py:380-395                           │
 │  → HTTP POST /update_weights_from_distributed       │
 └───────────────┬─────────────────────────────────────┘
                 │ HTTP
                 ▼
 ┌─────────────────────────────────────────────────────┐
 │  HTTP Server (http_server.py)                       │
 │  → tokenizer_manager.update_weights_from_distributed│
 └───────────────┬─────────────────────────────────────┘
                 │ async
                 ▼
 ┌─────────────────────────────────────────────────────┐
 │  TokenizerManager (communicator_mixin.py)           │
 │  async with model_update_lock.writer_lock:          │
 │    → dispatches to scheduler(s)                     │
 └───────────────┬─────────────────────────────────────┘
                 │ IPC (zmq)
                 ▼
 ┌─────────────────────────────────────────────────────┐
 │  Scheduler (scheduler_update_weights_mixin.py:73)   │
 │  → tp_worker.update_weights_from_distributed()      │
 └───────────────┬─────────────────────────────────────┘
                 │
                 ▼
 ┌─────────────────────────────────────────────────────┐
 │  ModelRunner (model_runner.py:1332-1389)            │
 │  ★ GPU WORK HAPPENS HERE ★                         │
 └─────────────────────────────────────────────────────┘
```

### The Critical Function: `ModelRunner.update_weights_from_distributed`

**File:** `sglang/python/sglang/srt/model_executor/model_runner.py:1332-1389`

```python
def update_weights_from_distributed(self, names, dtypes, shapes, group_name):
    group = self._model_update_group[group_name]

    # Step 1: Allocate empty GPU receive buffers
    weights = []
    handles = []
    for name, dtype, shape in zip(names, dtypes, shapes):
        target_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        weight = torch.empty(shape, dtype=target_dtype, device=self.device)  # ← GPU ALLOC
        handles.append(
            torch.distributed.broadcast(weight, src=0, group=group, async_op=True)
        )
        weights.append((name, weight))

    # Step 2: Wait for all NCCL broadcasts to complete
    for handle in handles:
        handle.wait()                    # ← Buffers now filled with trainer's weights

    # Step 3: Load into model (in-place copy)
    self.model.load_weights(weights)     # ← Copies into model's parameter storage
    return True, "Succeeded"
```

### NCCL Group Setup on Engine Side

**File:** `model_runner.py:1274-1317`

```python
def init_weights_update_group(self, master_address, master_port,
                               rank_offset, world_size, group_name, backend="nccl"):
    rank = rank_offset + self.tp_rank    # e.g., engine 0 TP rank 1 → rank 2
    self._model_update_group[group_name] = init_custom_process_group(
        backend="nccl",
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=rank,
        group_name=group_name,
    )
```

### In-Place Weight Loading

After NCCL broadcast fills the receive buffers, `model.load_weights(weights)` copies into existing parameter storage:

```
 In-place copy:
 ┌──────────────────────────────────────────────────┐
 │  param.data (existing model param on GPU)        │
 │  Address: 0x7f0000000000  [OLD weights]          │
 │                    ↑                              │
 │                    │ .copy_() via weight_loader   │
 │                    │ (GPU memcpy, same address)   │
 │                    │                              │
 │  loaded_weight (receive buffer from NCCL)        │
 │  Address: 0x7f0000100000  [NEW weights]          │
 │                                                  │
 │  After: param.data still at 0x7f0000000000       │
 │  but contents are now NEW weights.               │
 │  Receive buffer freed when ref count drops to 0. │
 └──────────────────────────────────────────────────┘
```

The model parameter's GPU address does not change — CUDA graphs, KV cache pointers, and other references remain valid.

### Quantized Weight Post-Processing

For quantized models (int4/Marlin), Miles patches SGLang with `post_process_weights`:

```
 Step 1: restore_weights_before_load (pre-process)
   Marlin-packed → GPTQ format (param.resize_(original_shape))

 Step 2: NCCL broadcast (weights arrive in GPTQ format)

 Step 3: process_weights_after_loading (post-process)
   GPTQ → Marlin-packed (gptq_marlin_moe_repack + marlin_moe_permute_scales)
```

---

## 9. GPU Memory Deep Dive: TP All-Gather

### Synchronous Per-Parameter (Megatron)

**File:** `backends/megatron_utils/update_weight/common.py:15-48`

```python
def all_gather_param(name, param):
    param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]  # alloc
    dist.all_gather(param_partitions, param.data, group=tp_group)               # sync
    param = torch.cat(param_partitions, dim=partition_dim)                       # concat
    return param
```

```
 GPU Memory States (TP=4, shard size S):
 ═══════════════════════════════════════════════════════

 Before all_gather:
 ┌──────────────────────────────────────────────────┐
 │  param.data (shard) .............. S bytes        │
 │  Total: S                                        │
 └──────────────────────────────────────────────────┘

 During all_gather:
 ┌──────────────────────────────────────────────────┐
 │  param.data (shard) .............. S bytes        │
 │  partitions[0..3] ................ 4×S bytes      │
 │  Total: 5S                                       │
 └──────────────────────────────────────────────────┘

 After torch.cat:
 ┌──────────────────────────────────────────────────┐
 │  param.data (shard) .............. S bytes        │
 │  partitions[0..3] ................ 4×S (pending GC│
 │  full_param ...................... 4×S bytes      │
 │  Peak: 9S  (before partitions GC'd)              │
 │  After GC: 5S (shard + full param)               │
 └──────────────────────────────────────────────────┘
```

### Async DTensor Redistribution (FSDP)

```python
param = param.redistribute(
    placements=[Replicate()] * param.device_mesh.ndim,
    async_op=True,   # ← starts all-gather, returns immediately
).to_local()         # ← returns AsyncCollectiveTensor
```

Multiple redistributions can be in-flight simultaneously within a bucket:

```
 Async overlap (FSDP, 3 params in bucket):
 ──────────────────────────────────────────────
 Time →

 NCCL stream:
   [redistribute p1][redistribute p2][redistribute p3]
   [──────── overlapped communication ────────────]

 Python thread:
   [append p1] [append p2] [append p3] → [wait_all]
   ↑ nearly instant (just starts async)

 Peak memory ≈ sum of all in-flight full params
```

### GLU Special Case

For `linear_fc1.weight` (gate-up fusion), partitions are rechunked after gather:

```python
param_partitions = [p.chunk(2, dim=0) for p in param_partitions]
param_partitions = [p[0] for p in param_partitions] + [p[1] for p in param_partitions]
```

This reorders `[gate_0, up_0, gate_1, up_1, ...]` → `[gate_0, gate_1, ..., up_0, up_1, ...]`.

---

## 10. GPU Memory Deep Dive: EP All-Gather (MoE)

Expert parameters go through **two gathering stages**: TP all-gather first, then EP all-gather.

**File:** `update_weight_from_distributed.py:183-219`

### EP All-Gather (Batched Async)

```python
# Phase 1: Launch ALL async all-gathers for K expert params
handles = []
for i, (_name, param) in enumerate(named_tensors):
    params = [torch.empty_like(param.data) for _ in range(ep_world_size)]  # alloc
    handle = dist.all_gather(params, param.data, group=ep_group, async_op=True)
    handles.append(handle)

# Phase 2: Wait for ALL at once (max parallelism)
for handle in handles:
    handle.wait()
```

```
 Peak GPU Memory During EP All-Gather
 ═══════════════════════════════════════════════════════

 For K buffered expert params, each E bytes post-TP:

 ┌──────────────────────────────────────────────────┐
 │  For each param_i (i = 1..K):                    │
 │                                                  │
 │  ┌────────┐ ┌────────┐     ┌────────┐           │
 │  │recv_buf│ │recv_buf│ ... │recv_buf│           │
 │  │ep_rk=0 │ │ep_rk=1 │     │ep_rk=7 │           │
 │  │ E bytes│ │ E bytes│     │ E bytes│           │
 │  └────────┘ └────────┘     └────────┘           │
 │                                                  │
 │  × K params = K × EP × E bytes total             │
 │  Plus K × E bytes (local expert shards)           │
 │                                                  │
 │  Peak: K × E × (EP + 1) bytes                    │
 │                                                  │
 │  Example: EP=8, K=10, E=500MB                    │
 │  → 10 × 500MB × 9 = 45 GB temporary!            │
 └──────────────────────────────────────────────────┘

 Buffer threshold divides by EP to prevent OOM:
 (buffer_size + param_size) × EP > update_weight_buffer_size
```

---

## 11. GPU Memory Deep Dive: Engine-Side

```
 Engine GPU Memory Timeline
 ═══════════════════════════════════════════════════════════════

 Phase 1: Normal inference
 ┌─────────────────────────────────────────────────────────┐
 │  ┌───────────────────┐  ┌────────────────────────────┐  │
 │  │  Model Weights (W) │  │  KV Cache (K)              │  │
 │  └───────────────────┘  └────────────────────────────┘  │
 └─────────────────────────────────────────────────────────┘

 Phase 2: After pause_generation + flush_cache
 ┌─────────────────────────────────────────────────────────┐
 │  ┌───────────────────┐                                  │
 │  │  Model Weights (W) │     ← KV cache freed            │
 │  └───────────────────┘                                  │
 │       [  Free GPU memory now available  ]               │
 └─────────────────────────────────────────────────────────┘

 Phase 3: Receive buffers allocated (one bucket)
 ┌─────────────────────────────────────────────────────────┐
 │  ┌───────────────────┐  ┌────────────────────────────┐  │
 │  │  Model Weights (W) │  │  Receive Buffers (B)       │  │
 │  │  [OLD values]      │  │  ≤ update_weight_buffer_sz │  │
 │  └───────────────────┘  └────────────────────────────┘  │
 │  Peak additional = B (bounded by trainer buffer_size)   │
 └─────────────────────────────────────────────────────────┘

 Phase 4: NCCL broadcast completes
 ┌─────────────────────────────────────────────────────────┐
 │  ┌───────────────────┐  ┌────────────────────────────┐  │
 │  │  Model Weights (W) │  │  Receive Buffers (B)       │  │
 │  │  [OLD values]      │  │  [FILLED with new weights] │  │
 │  └────────┬──────────┘  └─────────────┬──────────────┘  │
 │           │ ◄──── model.load_weights() ──── copy ──┘    │
 └─────────────────────────────────────────────────────────┘

 Phase 5: After load_weights, buffers freed
 ┌─────────────────────────────────────────────────────────┐
 │  ┌───────────────────┐                                  │
 │  │  Model Weights (W) │  ← [NEW values, in-place]       │
 │  └───────────────────┘                                  │
 │  Net memory change: 0 (same W bytes, new values)        │
 └─────────────────────────────────────────────────────────┘

 Phase 6: continue_generation → KV cache re-allocated
 ┌─────────────────────────────────────────────────────────┐
 │  ┌───────────────────┐  ┌────────────────────────────┐  │
 │  │  Model Weights (W) │  │  KV Cache (K) [re-alloc'd] │  │
 │  │  [NEW values]      │  │  [empty, ready for use]    │  │
 │  └───────────────────┘  └────────────────────────────┘  │
 └─────────────────────────────────────────────────────────┘
```

**Memory budget**: `W + B < total_GPU_memory`. Since KV cache is freed before sync, `B` uses that vacated space. The design is **memory-neutral** — engine footprint before and after sync is identical.

---

## 12. Async Overlap Analysis

### Where `async_op=True` Is Used

| Location | Operation | async? | Purpose |
|----------|-----------|--------|---------|
| Megatron TP all-gather (`common.py:35`) | `dist.all_gather` | **No** | Per-param, sequential |
| Megatron TP all-gather async (`common.py:81`) | `dist.all_gather` | **Yes** | Batched multi-param |
| Megatron EP all-gather (`update_weight_from_distributed.py:203`) | `dist.all_gather` | **Yes** | Overlap across experts |
| NCCL broadcast to engines (`update_weight_from_distributed.py:311`) | `dist.broadcast` | **Yes** | Overlap across bucket params |
| FSDP DTensor redistribute (`update_weight_utils.py:61-64`) | `.redistribute` | **Yes** | Overlap across bucket params |
| Engine-side NCCL receive (`model_runner.py:1368-1373`) | `dist.broadcast` | **Yes** | Overlap across bucket params |

### Megatron: No Cross-Bucket Pipelining

The non-expert path is **strictly sequential** between TP all-gather, HF conversion, and NCCL broadcast:

```
 Current (sequential per bucket):
 ──────────────────────────────────────────────────────────────

 Bucket 1:                              Bucket 2:
 [TP-gather][convert_hf][lock][bcast][unlock] → [TP-gather][convert_hf][lock][bcast][unlock]
                                              ↑
                              Cannot start until broadcast completes
```

```
 Hypothetical (double-buffered, NOT implemented):
 ──────────────────────────────────────────────────────────────

 Bucket 1:
 [TP-gather][convert_hf][lock][bcast][unlock]
                         [TP-gather B2][convert_hf B2][lock][bcast B2][unlock]
                         ↑ overlap gather with broadcast
```

### Within-Bucket Async Broadcast

Within a single bucket, both trainer and engine overlap N NCCL broadcasts:

```
 Trainer NCCL stream (within one bucket of N params):
 ──────────────────────────────────────────────────────

 [issue bcast p1][issue bcast p2]...[issue bcast pN]
 [────── NCCL internally pipelines all N ───────────]
                                                     [all done]

 Engine NCCL stream:
 [alloc+recv p1][alloc+recv p2]...[alloc+recv pN]
 [──────────── pipelined reception ─────────────]
                                                  [load_weights]
```

### EP All-Gather: True Async Batching

```
 K=4 expert params, EP=8:
 ──────────────────────────────────────────────────────
 NCCL stream (EP group):
   [all-gather expert_0]
         [all-gather expert_1]
               [all-gather expert_2]
                     [all-gather expert_3]
   [──── pipelined on NCCL stream ──────]
                                        [all complete]

 vs if synchronous:
   [expert_0]──[wait]──[expert_1]──[wait]──[expert_2]──[wait]──[expert_3]

 Speedup: ~K× reduction in wall time (bandwidth-bound)
```

### FSDP vs Megatron Comparison

```
 Stage                    │ FSDP              │ Megatron
 ─────────────────────────┼───────────────────┼──────────────────
 Param gathering          │ Async DTensor     │ Sync all_gather
 (within bucket)          │ (overlapped)      │ (sequential)
 ─────────────────────────┼───────────────────┼──────────────────
 Cross-bucket pipeline    │ No                │ No (+ lock)
 ─────────────────────────┼───────────────────┼──────────────────
 Broadcast (within bucket)│ async_op=True     │ async_op=True
 ─────────────────────────┼───────────────────┼──────────────────
 EP all-gather (experts)  │ N/A               │ async batched
```

### The Lock as a Pipeline Barrier

The `rollout_engine_lock` serializes broadcasts across PP stages:

```
 PP Stage 0:                          PP Stage 1:
 [acquire lock]                       [acquire lock — BLOCKED]
 [broadcast]                          [  waiting...  ]
 [release lock] ──────────────→       [lock acquired]
                                      [broadcast]
                                      [release lock]
```

This prevents NCCL deadlock when engines participate in multiple PP group broadcasts simultaneously.

### Rough Timing (70B model, TP=8, buffer=1GB)

| Phase | Time |
|-------|------|
| TP all-gather per param (NVLink) | ~10 ms |
| HF conversion (CPU) | ~1 ms |
| Lock acquire (Ray RPC) | ~5 ms |
| NCCL broadcast 1GB (IB) | ~200 ms |
| Engine load_weights (GPU memcpy) | ~20 ms |
| Lock release (Ray RPC) | ~5 ms |
| **Total per bucket** | **~250 ms** |
| **70B model (140GB bf16, 140 buckets)** | **~35 seconds** |

---

## 13. Key Files Reference

| File | Purpose |
|------|---------|
| `train.py` | Main training loop, calls `actor_model.update_weights()` |
| `miles/ray/actor_group.py` | `RayTrainGroup` — fans out `update_weights()` to all actor ranks |
| `miles/ray/rollout.py` | `RolloutManager` — creates engines, provides `get_rollout_engines_and_lock()` |
| `miles/ray/placement_group.py` | GPU allocation for non-colocated setup |
| `miles/backends/fsdp_utils/actor.py` | FSDP actor — selects weight updater, calls `update_weights()` |
| `miles/backends/fsdp_utils/update_weight_utils.py` | `UpdateWeightFromTensor` / `UpdateWeightFromDistributed` for FSDP |
| `miles/backends/megatron_utils/actor.py` | Megatron actor — selects weight updater, calls `update_weights()` |
| `miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py` | Non-colocated weight sync for Megatron |
| `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py` | Colocated weight sync for Megatron |
| `miles/backends/megatron_utils/update_weight/common.py` | `all_gather_param()`, `all_gather_params_async()`, `named_params_and_buffers()` |
| `miles/backends/megatron_utils/megatron_to_hf/__init__.py` | `convert_to_hf()` — Megatron → HuggingFace weight format conversion |
| `miles/backends/sglang_utils/sglang_engine.py` | `SGLangEngine` Ray actor — HTTP wrapper for SGLang |
| `sglang/.../model_executor/model_runner.py:1274-1420` | `init_weights_update_group()`, `update_weights_from_distributed()` — engine GPU-side |
| `sglang/.../managers/scheduler_update_weights_mixin.py` | Scheduler routing for weight updates |
| `sglang/.../managers/tp_worker.py:143-153` | TP worker forwarding to model runner |
| `docker/patch/latest/sglang.patch` | Miles patches to SGLang (post-processing, disaggregation memory) |

### Key Arguments

| Argument | Effect |
|----------|--------|
| `--colocate` | Switches to `UpdateWeightFromTensor` (Gloo IPC path) |
| `--rollout_num_gpus` | Number of GPUs for inference engines |
| `--rollout_num_gpus_per_engine` | GPUs per SGLang engine (defines NCCL group size) |
| `--update_weight_buffer_size` | Max bytes buffered before NCCL broadcast flush |
| `--offload_train` / `--offload_rollout` | Memory management during weight sync |
