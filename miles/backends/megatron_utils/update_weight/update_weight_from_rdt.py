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
        self.scheduler_actors = None
        self._self_handle = None
        self._tp_rank_views: dict[int, list[torch.Tensor]] = {}

        # Populated in connect_rollout_engines: schedulers grouped by TP rank
        self._schedulers_by_tp: dict[int, list] = {}
        self._shard_layouts: dict = {}  # tp_rank -> ParamLayout

        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
        )

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

    # --- Called by the train actor for RDT ---

    @torch.no_grad()
    def update_weights(self) -> None:
        """Weight sync: pause, iterate all params, shard per TP rank, transfer, resume."""
        self.weight_version += 1

        # Pause and flush engines
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

        # Source rank: create per-TP-rank sharding builders
        builders = None
        if self._is_pp_src_rank:
            builders = {
                tp_rank: RecipeShardedBucketBuilder(self._shard_layouts[tp_rank])
                for tp_rank in sorted(self._schedulers_by_tp.keys())
            }

        pbar = tqdm(desc="[RDT] Update weights") if self._is_pp_src_rank else None
        expert_params = []

        # Iterate all params: all-gather, convert to HF, shard per TP rank
        for name, param in named_params_and_buffers(self.args, self.model):
            gathered_param = all_gather_param(name, param)

            if ".experts." in name:
                expert_params.append((name, gathered_param))
                continue

            if self._is_pp_src_rank:
                for hf_name, tensor in convert_to_hf(
                    self.args, self.model_name, name, gathered_param,
                    self.quantization_config,
                ):
                    for builder in builders.values():
                        builder.add_hf_tensor(hf_name, tensor)
            if pbar is not None:
                pbar.update(1)

        # Expert params: EP all-gather + convert + shard
        dist.barrier(group=get_gloo_group())
        if expert_params:
            hf_tensors = self._gather_expert_params(expert_params)
            if self._is_pp_src_rank and hf_tensors:
                for hf_name, tensor in hf_tensors:
                    for builder in builders.values():
                        builder.add_hf_tensor(hf_name, tensor)
            if pbar is not None:
                pbar.update(len(expert_params))
        dist.barrier(group=get_gloo_group())

        # Flush builders -> contiguous sharded tensors, then transfer
        if self._is_pp_src_rank:
            self._tp_rank_views = {}
            tp_param_names = {}
            for tp_rank, builder in builders.items():
                presharded = builder.build()
                tp_param_names[tp_rank] = [name for name, _ in presharded]
                self._tp_rank_views[tp_rank] = [t.contiguous() for _, t in presharded]

            # Transfer to all scheduler actors
            all_refs = []
            for tp_rank, actors in self._schedulers_by_tp.items():
                param_names = tp_param_names[tp_rank]
                for a in actors:
                    all_refs.append(
                        a.pull_weights.remote(self._self_handle, param_names, tp_rank)
                    )
            ray.get(all_refs)
            self._tp_rank_views = {}

        if pbar is not None:
            pbar.close()

        # Resume engines
        self.resume_engines()
