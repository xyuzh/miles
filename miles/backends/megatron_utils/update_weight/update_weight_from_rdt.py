"""
Update SchedulerActor engines via RDT/NIXL.
Trainer all-gathers params, converts to HF format, shards per TP rank
using recipes from sglang, and exports via @ray.method(tensor_transport="nixl").
SchedulerActors pull concurrently via RDMA into model param buffers.

Lifecycle:
    1. Pause engines, flush cache
    2. Iterate all params: all-gather -> convert to HF -> shard per TP rank
    3. Transfer sharded tensors to all scheduler actors via RDT
    4. Resume engines, post-process quantization
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
    """Builds pre-sharded weight tensors using sharding recipes from sglang.

    Each recipe describes exactly which narrow() ops to apply to a full HF
    tensor to produce the shard for a specific TP rank.  Stacked params
    (QKV, gate_up) are accumulated by concat_group and fused once all
    components arrive.
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

    def _accumulate_stacked(self, recipe: ParamShardingRecipe, sharded: torch.Tensor) -> None:
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
        assert not self._stacked_accum, f"Incomplete stacked params: {list(self._stacked_accum.keys())}"
        result = self._result
        self._result = []
        return result


class UpdateWeightFromRDT:
    """
    Update SchedulerActor engines via RDT/NIXL. No NCCL groups needed.
    Iterates all params, all-gathers, converts to HF, shards per TP rank,
    then transfers all sharded tensors to rollout engines in one shot.
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
        self.scheduler_actors: list[ActorHandle] | None = None
        self._tp_rank_views: dict[int, list[torch.Tensor]] = {}

        # Populated in connect_rollout_engines: schedulers grouped by TP rank
        self._schedulers_by_tp: dict[int, list[ActorHandle]] = {}
        self._shard_layouts: dict[int, ParamLayout] = {}

        self._is_tp_rank_zero = mpu.get_tensor_model_parallel_rank() == 0

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle | None = None,
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
        dist.barrier(group=get_gloo_group())

    # --- Called by the train actor for RDT ---

    def _feed_hf_tensors_to_builders(
        self,
        builders: dict[int, RecipeShardedBucketBuilder],
        hf_tensors: list[tuple[str, torch.Tensor]],
    ) -> None:
        for hf_name, tensor in hf_tensors:
            for builder in builders.values():
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
        dist.barrier(group=get_gloo_group())

    def _gather_and_shard_all_params(self) -> dict[int, RecipeShardedBucketBuilder] | None:
        builders = None
        if self._is_tp_rank_zero:
            builders = {
                tp_rank: RecipeShardedBucketBuilder(self._shard_layouts[tp_rank])
                for tp_rank in sorted(self._schedulers_by_tp.keys())
            }

        pbar = tqdm(desc="[RDT] Update weights") if self._is_tp_rank_zero else None

        for name, param in named_params_and_buffers(self.args, self.model):
            gathered = all_gather_param(self.args, name, param)

            if builders is not None:
                self._feed_hf_tensors_to_builders(
                    builders,
                    convert_to_hf(self.args, self.model_name, name, gathered, self.quantization_config),
                )
            if pbar is not None:
                pbar.update(1)

        dist.barrier(group=get_gloo_group())

        if pbar is not None:
            pbar.close()
        return builders

    def _transfer_to_schedulers(self, builders: dict[int, RecipeShardedBucketBuilder]) -> None:
        self._tp_rank_views = {}
        tp_param_names = {}
        for tp_rank, builder in builders.items():
            presharded = builder.build()
            tp_param_names[tp_rank] = [name for name, _ in presharded]
            self._tp_rank_views[tp_rank] = [t.contiguous() for _, t in presharded]

        all_refs = []
        for tp_rank, actors in self._schedulers_by_tp.items():
            weights_ref = ray.put(self._tp_rank_views[tp_rank], _tensor_transport="nixl")
            for actor in actors:
                all_refs.append(actor.pull_weights.remote([weights_ref], tp_param_names[tp_rank]))
        ray.get(all_refs)
        self._tp_rank_views = {}

    @torch.no_grad()
    def update_weights(self) -> None:
        """Weight sync: pause, gather+shard all params, transfer via RDT, resume."""
        self.weight_version += 1
        self._pause_engines()
        builders = self._gather_and_shard_all_params()
        if builders is not None:
            self._transfer_to_schedulers(builders)
        self.resume_engines()
