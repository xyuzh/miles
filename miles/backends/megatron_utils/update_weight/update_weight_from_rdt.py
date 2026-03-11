"""
Update SchedulerActor engines via RDT/NIXL.
Trainer builds FlattenedTensorBucket, exports via @ray.method(tensor_transport="nixl"),
SchedulerActors pull concurrently.

Lifecycle (bucket-by-bucket, matching NCCL path):
    1. start_weight_sync()      — pause, flush, init param iterator
    2. prepare_next_bucket()    — gather params until buffer_size exceeded, build flat tensor
       (driver calls export_weights_rdt → RDT transfer → engines receive)
       cleanup_bucket()         — free flat tensor
       ... repeat 2 until no more params ...
    3. finish_weight_sync()     — resume engines, post-process quantization
"""

from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from tqdm import tqdm

from miles.utils.distributed_utils import get_gloo_group

from ..megatron_to_hf import convert_to_hf
from .common import all_gather_param, named_params_and_buffers
from .update_weight_from_distributed import post_process_weights


class CudaRawBuffer:
    """GPU buffer allocated via cudaMalloc (bypasses PyTorch caching allocator).

    Exposes __cuda_array_interface__ so torch.as_tensor() can create a
    tensor viewing this memory without PyTorch taking ownership.
    This is needed because NIXL/GDR copy cannot pin memory allocated via
    cuMemCreate (used by PyTorch's expandable_segments mode).
    """

    def __init__(self, size_bytes: int):
        if size_bytes <= 0:
            raise ValueError(
                f"CudaRawBuffer requires positive size, got {size_bytes}. "
                "This likely means no tensors matched sharding recipes."
            )
        from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
        self._lib = CudaRTLibrary()
        self._ptr = self._lib.cudaMalloc(size_bytes)
        self._size = size_bytes
        self.__cuda_array_interface__ = {
            "shape": (size_bytes,),
            "typestr": "|u1",  # uint8
            "data": (self._ptr.value, False),  # (address, read_only)
            "version": 3,
        }

    def as_tensor(self) -> torch.Tensor:
        """Return a torch.Tensor (uint8) viewing this buffer."""
        return torch.as_tensor(self, device=f"cuda:{torch.cuda.current_device()}")

    def free(self):
        if self._ptr is not None:
            self._lib.cudaFree(self._ptr)
            self._ptr = None

    def __del__(self):
        self.free()


class RecipeShardedBucketBuilder:
    """Builds pre-sharded weight tensors using sharding recipes from sglang.

    Each recipe describes exactly which narrow() ops to apply to a full HF
    tensor to produce the shard for a specific TP rank.  Stacked params
    (QKV, gate_up) are accumulated by concat_group and fused once all
    components arrive.
    """

    def __init__(self, layout):
        from sglang.srt.ray.weight_sync import ParamLayout

        self.layout: ParamLayout = layout
        self.recipes = layout.sharding_recipes

        # Precompute concat groups: internal_name -> sorted list of concat_orders
        self._concat_groups: dict[str, list[int]] = {}
        for recipe in self.recipes.values():
            if recipe.concat_group is not None:
                self._concat_groups.setdefault(recipe.concat_group, []).append(
                    recipe.concat_order
                )
        for group in self._concat_groups.values():
            group.sort()

        # Accumulator for stacked params: concat_group -> {concat_order: tensor}
        self._stacked_accum: dict[str, dict[int, torch.Tensor]] = {}

        self._result: list[tuple[str, torch.Tensor]] = []

    def add_hf_tensor(self, hf_name: str, tensor: torch.Tensor) -> None:
        """Process one HF-named tensor, sharding it per the recipe."""
        recipe = self.recipes.get(hf_name)
        if recipe is None:
            return

        sharded = tensor
        for op in recipe.narrow_ops:
            sharded = sharded.narrow(op.dim, op.start, op.length)

        if recipe.concat_group is not None:
            self._accumulate_stacked(recipe, sharded)
        else:
            self._result.append((recipe.internal_name, sharded))

    def _accumulate_stacked(self, recipe, sharded: torch.Tensor) -> None:
        group = recipe.concat_group
        if group not in self._stacked_accum:
            self._stacked_accum[group] = {}

        self._stacked_accum[group][recipe.concat_order] = sharded

        expected_orders = self._concat_groups[group]
        if len(self._stacked_accum[group]) == len(expected_orders):
            ordered = [self._stacked_accum[group][o] for o in expected_orders]
            fused = torch.cat(ordered, dim=0)
            self._result.append((recipe.internal_name, fused))
            del self._stacked_accum[group]

    def build(self) -> list[tuple[str, torch.Tensor]]:
        assert not self._stacked_accum, (
            f"Incomplete stacked params: {list(self._stacked_accum.keys())}"
        )
        result = self._result
        self._result = []
        return result


class UpdateWeightFromRDT:
    """
    Update SchedulerActor engines via RDT/NIXL. No NCCL groups needed.
    Weight preparation is separated from distribution — the driver orchestrates
    the RDT export after each bucket is prepared.

    Uses buffer-size-based bucketing (args.update_weight_buffer_size) to mirror
    the NCCL path's memory-efficient bucket-by-bucket approach.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.scheduler_actors = None
        self._self_handle = None
        self._current_metadata: list | None = None
        self._current_hf_tensors: list[tuple[str, torch.Tensor]] | None = None
        self._tp_rank_views: dict[int, list[torch.Tensor]] = {}
        self._persistent_buffer: CudaRawBuffer | None = None
        self._persistent_tensor: torch.Tensor | None = None

        # Populated in connect_rollout_engines: schedulers grouped by TP rank
        self._schedulers_by_tp: dict[int, list] = {}
        self._shard_layouts: dict = {}  # tp_rank -> ParamLayout

        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
        )

        # Stateful iterator state for bucket-by-bucket preparation
        self._param_iter = None
        self._phase = "idle"  # "non_expert" | "expert" | "done"
        self._pbar: tqdm | None = None
        # Carry-over param that exceeded the previous bucket's budget
        self._pending_param: tuple[str, torch.Tensor] | None = None
        # Accumulated expert params for the current expert bucket
        self._expert_buffer: list[tuple[str, torch.Tensor]] = []
        self._expert_buffer_size: int = 0

    def set_actor_handle(self, handle) -> None:
        """Store the trainer's own actor handle for scheduler pull_weights calls."""
        self._self_handle = handle

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence,
        rollout_engine_lock=None,
    ) -> None:
        """Store rollout engines and extract SchedulerActor handles for RDT."""
        self.rollout_engines = rollout_engines
        self.scheduler_actors = []
        for engine in rollout_engines:
            actors = ray.get(engine.get_scheduler_actors.remote())
            self.scheduler_actors.extend(actors)

        # Fetch shard configs and group by TP rank (one-time setup)
        if self.scheduler_actors:
            layouts = ray.get([a.get_param_layout.remote() for a in self.scheduler_actors])
            self._schedulers_by_tp = {}
            self._shard_layouts = {}
            for actor, layout in zip(self.scheduler_actors, layouts):
                tp_rank = layout.tp_rank
                self._schedulers_by_tp.setdefault(tp_rank, []).append(actor)
                if tp_rank not in self._shard_layouts:
                    self._shard_layouts[tp_rank] = layout

    @torch.no_grad()
    def start_weight_sync(self) -> None:
        """All ranks: pause/flush engines, init param iterator, pre-process quant."""
        self.weight_version += 1

        if dist.get_rank() == 0:
            ray.get([e.pause_generation.remote() for e in self.rollout_engines])
            ray.get([e.flush_cache.remote() for e in self.rollout_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in [
                "compressed-tensors"
            ]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        self._param_iter = iter(named_params_and_buffers(self.args, self.model))
        self._phase = "non_expert"
        self._pending_param = None
        self._expert_buffer = []
        self._expert_buffer_size = 0
        self._pbar = (
            tqdm(desc="[RDT] Update weights", total=0)
            if self._is_pp_src_rank
            else None
        )

    @torch.no_grad()
    def prepare_next_bucket(self) -> bool:
        """All ranks: gather params until buffer_size exceeded, build flat tensor.

        Returns True if a bucket was prepared, False if no more params.
        All TP ranks call the same params in the same order and track buffer_size
        identically (using post-all_gather param sizes), so all ranks break at
        the same bucket boundaries.
        """
        if self._phase == "non_expert":
            result = self._prepare_non_expert_bucket()
            if result:
                return True
            # Non-expert phase exhausted; barrier before expert phase
            dist.barrier(group=get_gloo_group())

        if self._phase == "expert":
            result = self._prepare_expert_bucket()
            if result:
                return True

        if self._phase == "done":
            dist.barrier(group=get_gloo_group())

        return False

    def _prepare_non_expert_bucket(self) -> bool:
        """Gather non-expert params into a bucket. Returns True if bucket built.

        All ranks track buffer_size identically (using the all-gathered param size)
        so they break at the same bucket boundary. Only source rank does HF convert
        and builds the flat tensor.
        """
        hf_buffer: list[tuple[str, torch.Tensor]] = []
        buffer_size = 0

        # Process carry-over param from previous bucket
        if self._pending_param is not None:
            name, gathered_param = self._pending_param
            self._pending_param = None
            param_size = gathered_param.numel() * gathered_param.element_size()
            if self._is_pp_src_rank:
                hf_buffer.extend(convert_to_hf(
                    self.args, self.model_name, name, gathered_param,
                    self.quantization_config,
                ))
            buffer_size += param_size

        for name, param in self._param_iter:
            if ".experts." in name:
                # First expert param: TP all-gather it, save for expert phase
                gathered_param = all_gather_param(name, param)
                self._expert_buffer.append((name, gathered_param))
                param_size = gathered_param.numel() * gathered_param.element_size()
                self._expert_buffer_size += param_size
                self._phase = "expert"
                break

            gathered_param = all_gather_param(name, param)
            param_size = gathered_param.numel() * gathered_param.element_size()

            # Check buffer threshold (same logic as NCCL path)
            if buffer_size + param_size > self.args.update_weight_buffer_size and buffer_size > 0:
                # Save this param for the next bucket
                self._pending_param = (name, gathered_param)
                self._build_bucket(hf_buffer)
                return True

            if self._is_pp_src_rank:
                hf_buffer.extend(convert_to_hf(
                    self.args, self.model_name, name, gathered_param,
                    self.quantization_config,
                ))
            buffer_size += param_size
        else:
            # Iterator exhausted without hitting experts
            if self._phase == "non_expert":
                self._phase = "done"

        if buffer_size > 0:
            # Source rank built the bucket; non-source must also return True
            # so that all ranks stay in sync (same number of barrier calls).
            self._build_bucket(hf_buffer)  # no-op on non-source rank
            return True

        return False

    def _prepare_expert_bucket(self) -> bool:
        """Gather expert params into buckets using EP-size threshold.

        Returns True if bucket built. All ranks track buffer_size identically.
        """
        ep_size = mpu.get_expert_model_parallel_world_size()

        for name, param in self._param_iter:
            assert ".experts." in name, f"Expected expert param, got {name}"
            gathered_param = all_gather_param(name, param)
            param_size = gathered_param.numel() * gathered_param.element_size()

            # Expert threshold: multiply by EP size (matching NCCL path)
            if (
                self._expert_buffer_size + param_size
            ) * ep_size > self.args.update_weight_buffer_size and self._expert_buffer:
                # Flush current expert buffer
                hf_tensors = self._gather_expert_params(self._expert_buffer)
                self._expert_buffer = [(name, gathered_param)]
                self._expert_buffer_size = param_size
                if hf_tensors:
                    self._build_bucket(hf_tensors)
                    return True
                # If no hf_tensors (non-source rank), still report bucket ready
                # since source rank did build one
                return True

            self._expert_buffer.append((name, gathered_param))
            self._expert_buffer_size += param_size

        # Flush remaining expert buffer
        if self._expert_buffer:
            hf_tensors = self._gather_expert_params(self._expert_buffer)
            self._expert_buffer = []
            self._expert_buffer_size = 0
            self._phase = "done"
            if hf_tensors:
                self._build_bucket(hf_tensors)
                return True
            # Non-source ranks: source still built a bucket
            return True

        self._phase = "done"
        return False

    def _ensure_export_buffer(self, min_size: int) -> None:
        """Ensure the persistent cudaMalloc buffer is at least min_size bytes.

        On first call (or if the buffer is too small), allocates a new buffer
        via cudaMalloc and pre-registers it with NIXL. The buffer persists across
        sync cycles for zero-overhead reuse.
        """
        if min_size <= 0:
            return
        if self._persistent_buffer is not None and self._persistent_buffer._size >= min_size:
            return  # Already big enough

        # Old buffer (if any) leaks until process death — acceptable since
        # register_nixl_memory registrations are never deregistered (by design).
        self._persistent_buffer = CudaRawBuffer(min_size)
        self._persistent_tensor = self._persistent_buffer.as_tensor()

        # Pre-register with NIXL so subsequent exports get cache hits.
        ray.experimental.register_nixl_memory(self._persistent_tensor)

    def _build_bucket(self, hf_named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        """Source rank: save HF tensors for per-TP sharding (packing happens later)."""
        if not self._is_pp_src_rank:
            return

        self._current_hf_tensors = hf_named_tensors

        if self._pbar is not None:
            self._pbar.update(1)

    def _shard_and_pack_all_tp_ranks(self) -> dict[int, list[str]]:
        """Shard HF tensors for ALL TP ranks, pack into single flat buffer.

        Returns dict mapping tp_rank -> list of param names.
        Populates self._tp_rank_views with per-TP-rank tensor view lists.
        """
        all_named_tensors: list[tuple[str, torch.Tensor]] = []
        tp_boundaries: dict[int, tuple[int, int]] = {}  # tp_rank -> (start_index, count)

        for tp_rank in sorted(self._schedulers_by_tp.keys()):
            layout = self._shard_layouts[tp_rank]
            builder = RecipeShardedBucketBuilder(layout)
            for name, tensor in self._current_hf_tensors:
                builder.add_hf_tensor(name, tensor)
            presharded = builder.build()

            start = len(all_named_tensors)
            all_named_tensors.extend(presharded)
            tp_boundaries[tp_rank] = (start, len(presharded))

        self._pack_into_flat_tensor(all_named_tensors)

        self._tp_rank_views = {}
        tp_param_names: dict[int, list[str]] = {}
        for tp_rank, (start, count) in tp_boundaries.items():
            views = []
            names = []
            for i in range(start, start + count):
                m = self._current_metadata[i]
                view = self._persistent_tensor[m.start_idx:m.end_idx].view(m.dtype).reshape(m.shape)
                views.append(view)
                names.append(m.name)
            self._tp_rank_views[tp_rank] = views
            tp_param_names[tp_rank] = names

        return tp_param_names

    def _pack_into_flat_tensor(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
    ) -> None:
        """Pack named tensors into the persistent cudaMalloc buffer."""
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorMetadata

        torch.cuda.empty_cache()

        # Compute metadata and total byte size
        metadata = []
        current_idx = 0
        for name, tensor in named_tensors:
            byte_numel = tensor.numel() * tensor.element_size()
            metadata.append(FlattenedTensorMetadata(
                name=name,
                shape=tensor.shape,
                dtype=tensor.dtype,
                start_idx=current_idx,
                end_idx=current_idx + byte_numel,
                numel=byte_numel,
            ))
            current_idx += byte_numel

        # Ensure persistent cudaMalloc buffer is large enough
        self._ensure_export_buffer(current_idx)

        # Copy tensors into the persistent buffer, then take a view
        # of exactly the needed size. The view shares the same
        # untyped_storage().data_ptr() → NIXL cache hit, no re-registration.
        for i, (_name, tensor) in enumerate(named_tensors):
            m = metadata[i]
            self._persistent_tensor[m.start_idx:m.end_idx].copy_(
                tensor.flatten().view(torch.uint8)
            )
        self._current_metadata = metadata

    def _gather_expert_params(
        self,
        named_tensors: list[tuple[str, torch.Tensor]],
    ) -> list[tuple[str, torch.Tensor]]:
        """EP all-gather + HF convert for expert parameters."""
        names = [name for name, _ in named_tensors]
        all_names = [None] * mpu.get_expert_model_parallel_world_size()
        dist.all_gather_object(
            all_names, names, group=mpu.get_expert_model_parallel_group()
        )

        all_gathered_params = [
            [] for _ in range(mpu.get_expert_model_parallel_world_size())
        ]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(
                params,
                param.data,
                group=mpu.get_expert_model_parallel_group(),
                async_op=True,
            )
            handles.append(handle)
            for ep_rank, ep_names in enumerate(all_names):
                all_gathered_params[ep_rank].append((ep_names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()

        if not self._is_pp_src_rank:
            return []

        all_gathered_params_flat = sum(all_gathered_params, [])
        hf_tensors = []
        for name, param in all_gathered_params_flat:
            hf_tensors += convert_to_hf(
                self.args,
                self.model_name,
                name,
                param,
                self.quantization_config,
            )
        return hf_tensors

    @torch.no_grad()
    def finish_weight_sync(self) -> None:
        """All ranks: resume engines, post-process quantization, close pbar."""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        self._param_iter = None
        self._phase = "idle"
        self.resume_engines()

    @torch.no_grad()
    def resume_engines(self) -> None:
        """Resume generation + post-process quantization."""
        if dist.get_rank() == 0:
            ray.get([e.continue_generation.remote() for e in self.rollout_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in [
                "compressed-tensors",
                "mxfp8",
            ]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

    def cleanup_bucket(self) -> None:
        """Clear refs to current bucket. Persistent buffer stays alive."""
        self._current_metadata = None
        self._current_hf_tensors = None
        self._tp_rank_views = {}

    # --- Called by the train actor for RDT ---

    @torch.no_grad()
    def update_weights(self) -> None:
        """Self-contained weight sync: pause, bucket loop, resume.

        The source rank (DP=0, TP=0) drives the bucket loop. For each bucket,
        it shards HF tensors per TP rank and tells schedulers to pull via RDT.
        The schedulers call trainer.export_weights_rdt.remote() which runs in
        the rdt_export concurrency group (so it doesn't deadlock with this task).
        """
        self.start_weight_sync()
        while True:
            if not self.prepare_next_bucket():
                break
            if self._is_pp_src_rank:
                tp_param_names = self._shard_and_pack_all_tp_ranks()
                all_refs = []
                for tp_rank, actors in self._schedulers_by_tp.items():
                    param_names = tp_param_names[tp_rank]
                    for a in actors:
                        all_refs.append(
                            a.pull_weights.remote(self._self_handle, param_names, tp_rank)
                        )
                ray.get(all_refs)
            self.cleanup_bucket()
        self.finish_weight_sync()
