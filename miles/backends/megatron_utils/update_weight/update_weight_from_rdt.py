"""
Update SchedulerActor engines via RDT/NIXL (assumes inference TP=1).
Trainer all-gathers params, converts to HF format, assembles stacked
params using recipes from sglang, and exports via RDT/NIXL.
SchedulerActors pull concurrently via RDMA into model param buffers.

Lifecycle & memory:

    1. Pause engines, flush cache
       No tensor work. Rank 0 sends Ray RPCs; all ranks barrier.

    2. Iterate all params: all-gather → convert to HF → assemble → batch transfer
       Per param, on every TP rank:
         all_gather_param   – alloc tp_size buffers, NCCL fills them [copy]
                            – torch.cat partitions → full tensor   [copy]
       On tp_rank_zero only:
         convert_to_hf      – remove_padding: view (no copy)
                            – model-specific transform: may split/transpose [copy, model-dependent]
                            – quantize_params: new dtype tensor    [copy if enabled]
         assemble stacked   – accumulate refs per concat_group (no copy)
                            – torch.cat fuse when group complete   [copy]
         batch transfer     – when accumulated HF tensors >= _BATCH_BYTES,
                              flush via RDT: .contiguous() [copy if needed],
                              ray.put/NIXL, scheduler RDMA pull [network copy].
                              Transfer is synchronous (blocks until all schedulers
                              ack), so it does NOT overlap with the next all-gather.
       After loop, transfer any remaining params.
       Peak on tp_rank_zero: up to _BATCH_BYTES of accumulated HF tensors
       + one full-size gathered param and its conversion intermediates.
       Non-tp-rank-zero: one gathered param (immediately freed).

    3. Resume engines, post-process quantization
       No tensor work. Rank 0 sends Ray RPCs; all ranks barrier.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle
from tqdm import tqdm

from miles.utils.distributed_utils import get_gloo_group

from ..megatron_to_hf import convert_to_hf
from .common import all_gather_param, named_params_and_buffers
from .update_weight_from_distributed import post_process_weights

if TYPE_CHECKING:
    from sglang.srt.ray.weight_sync import ParamLayout, ParamShardingRecipe


class RecipeShardedBucketBuilder:
    """Assembles internal model params from HF tensors using sharding recipes.

    With tp=1, no narrow/slicing is needed.  Stacked params (QKV, gate_up)
    are accumulated by concat_group and fused once all components arrive.
    """

    def __init__(self, layout: ParamLayout) -> None:
        self.layout = layout
        self.recipes = layout.sharding_recipes

        # Precompute concat groups: internal_name -> sorted list of concat_orders
        self._concat_groups: dict[str, list[int]] = {}
        for recipe in self.recipes.values():
            if recipe.concat_group is not None:
                self._concat_groups.setdefault(recipe.concat_group, []).append(recipe.concat_order)
        for group in self._concat_groups.values():
            group.sort()

        # Accumulator for stacked params: concat_group -> {concat_order: tensor}
        self._stacked_accum: dict[str, dict[int, torch.Tensor]] = {}

        self._result: list[tuple[str, torch.Tensor]] = []

    def add_hf_tensor(self, hf_name: str, tensor: torch.Tensor) -> None:
        """Process one HF-named tensor per the recipe (tp=1, no narrow ops)."""
        recipe = self.recipes.get(hf_name)
        if recipe is None:
            return

        if recipe.concat_group is not None:
            self._accumulate_stacked(recipe, tensor)
        else:
            self._result.append((recipe.internal_name, tensor))

    def _accumulate_stacked(self, recipe: ParamShardingRecipe, sharded: torch.Tensor) -> None:
        """Accumulate stacked param components and fuse when the group is complete.

        Memory:
            - Storing sharded ref per concat_order: no copy (keeps existing tensor alive).
            - torch.cat when group complete: [copy] — allocates a new contiguous tensor.
            - del accum entry: frees individual component refs (but tensors may
              stay alive if referenced elsewhere until the fused tensor is consumed).
        """
        group = recipe.concat_group
        if group not in self._stacked_accum:
            self._stacked_accum[group] = {}

        self._stacked_accum[group][recipe.concat_order] = sharded  # no copy

        expected_orders = self._concat_groups[group]
        if len(self._stacked_accum[group]) == len(expected_orders):
            ordered = [self._stacked_accum[group][o] for o in expected_orders]
            fused = torch.cat(ordered, dim=0)  # [copy]
            self._result.append((recipe.internal_name, fused))
            del self._stacked_accum[group]

    def flush(self) -> list[tuple[str, torch.Tensor]]:
        """Return and clear completed params (safe to call mid-iteration)."""
        result = self._result
        self._result = []
        return result

    def accumulated_bytes(self) -> int:
        """Total bytes of completed params waiting to be flushed."""
        return sum(t.nelement() * t.element_size() for _, t in self._result)

    def build(self) -> list[tuple[str, torch.Tensor]]:
        assert not self._stacked_accum, f"Incomplete stacked params: {list(self._stacked_accum.keys())}"
        return self.flush()


class UpdateWeightFromRDT:
    """
    Update SchedulerActor engines via RDT/NIXL. No NCCL groups needed.
    Iterates all params, all-gathers, converts to HF, assembles via recipes,
    then streams assembled tensors to rollout engines in batches.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        is_lora: bool = False,
    ) -> None:
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._scheduler_actors: list[ActorHandle] = []
        self._layout: ParamLayout | None = None
        self._tensor_views: list[torch.Tensor] = []

        self._is_tp_rank_zero = mpu.get_tensor_model_parallel_rank() == 0

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None = None,
    ) -> None:
        """Store rollout engines and extract SchedulerActor handles for RDT."""
        self.rollout_engines = rollout_engines
        self._scheduler_actors = []
        for engine in rollout_engines:
            actors = ray.get(engine.get_scheduler_actors.remote())
            self._scheduler_actors.extend(actors)

        # Fetch layout from the first scheduler (all are identical with tp=1)
        if self._scheduler_actors:
            self._layout = ray.get(self._scheduler_actors[0].get_param_layout.remote())

    @torch.no_grad()
    def resume_engines(self) -> None:
        """Post-process quantization, then resume generation."""
        if dist.get_rank() == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in [
                "compressed-tensors",
                "mxfp8",
            ]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([e.continue_generation.remote() for e in self.rollout_engines])
        # Barrier: all TP ranks wait until rank 0 finishes resuming engines.
        dist.barrier(group=get_gloo_group())

    # --- Called by the train actor for RDT ---

    def _feed_hf_tensors_to_builder(
        self,
        builder: RecipeShardedBucketBuilder,
        hf_tensors: list[tuple[str, torch.Tensor]],
    ) -> None:
        for hf_name, tensor in hf_tensors:
            builder.add_hf_tensor(hf_name, tensor)

    def _pause_engines(self) -> None:
        if dist.get_rank() == 0:
            ray.get([e.pause_generation.remote() for e in self.rollout_engines])
            ray.get([e.flush_cache.remote() for e in self.rollout_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        # Barrier: all TP ranks wait until rank 0 finishes pausing engines.
        dist.barrier(group=get_gloo_group())

    # 4 GiB batch threshold — keeps peak GPU memory bounded
    _BATCH_BYTES = 4 * 1024**3

    def _transfer_batch(self, batch: list[tuple[str, torch.Tensor]]) -> None:
        """Transfer one batch of (name, tensor) pairs to all scheduler actors."""
        if not batch:
            return
        param_names = [name for name, _ in batch]
        tensor_views = [t.contiguous() for _, t in batch]

        weights_ref = ray.put(tensor_views, _tensor_transport="nixl")
        all_refs = []
        for actor in self._scheduler_actors:
            all_refs.append(actor.pull_weights.remote([weights_ref], param_names))
        ray.get(all_refs)

    @torch.no_grad()
    def update_weights(self) -> None:
        """Weight sync: pause, gather+shard params, stream via RDT in batches, resume."""
        self.weight_version += 1
        self._pause_engines()

        builder = None
        if self._is_tp_rank_zero:
            builder = RecipeShardedBucketBuilder(self._layout)

        pbar = tqdm(desc="[RDT] Update weights") if self._is_tp_rank_zero else None

        for name, param in named_params_and_buffers(self.args, self.model):
            gathered = all_gather_param(self.args, name, param)

            if builder is not None:
                hf_tensors = convert_to_hf(self.args, self.model_name, name, gathered, self.quantization_config)
                del gathered  # free all-gather temp memory before accumulating
                self._feed_hf_tensors_to_builder(builder, hf_tensors)
                del hf_tensors
                # Flush when accumulated batch is large enough to bound GPU memory
                if builder.accumulated_bytes() >= self._BATCH_BYTES:
                    self._transfer_batch(builder.flush())
            else:
                del gathered
            if pbar is not None:
                pbar.update(1)

        # Transfer remaining params
        if builder is not None:
            self._transfer_batch(builder.build())

        # Barrier: non-tp-rank-zero ranks skip builder/transfer entirely,
        # so they wait here until rank 0 finishes all RDT transfers.
        dist.barrier(group=get_gloo_group())

        if pbar is not None:
            pbar.close()
        self.resume_engines()
