# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

import torch.nn.functional as F
import gc
import os
import numpy as np
# from auto_gptq.nn_modules.qlinear.qlinear_cuda import QuantLinear
# from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear as QuantLinearOld

import torch
import functools
import json
import time
from dataclasses import dataclass, field

from tqdm import tqdm

from finemoe.ops.op_builder.prefetch import PrefetchBuilder
from finemoe.models import (
    SyncQwen2MoeSparseMoeBlock,
    Qwen2MoeMLP,
    SyncOlmoeSparseMoeBlock,
    OlmoeMLP,
    SyncMixtralSparseMoeBlock,
    SyncDeepseekV2Moe,
    SyncDeepseekV3MoE,
)
from finemoe.utils import ArcherConfig
from finemoe.utils.arguments import copy_args_to_device, copy_kwargs_to_device

from finemoe.memory import ExpertPrefetcher
import finemoe
from finemoe.utils import (
    parse_moe_param,
    parse_expert_layout,
    parse_expert_id,
    get_packed_expert_schema,
    parse_expert_dtype,
    parse_expert_dtype_id,
    parse_packed_expert_tensor,
    expand_state_dict_for_offload,
)
from finemoe.common import parse_expert_type
from finemoe.memory import ExpertTracer

from typing import Dict, Type, Union
from transformers import (
    AutoConfig,
)
from transformers.modeling_utils import PreTrainedModel
try:
    from transformers.modeling_utils import PretrainedConfig
except ImportError:
    from transformers import PretrainedConfig
import transformers
from typing import Callable

from safetensors import safe_open

import re

use_jit = False
try:
    import finemoe.ops.prefetch.prefetch_op as prefetch_op
except ImportError:
    print(f"Do not detect pre-installed ops, use JIT mode")
    use_jit = True


# class ArcherException(Exception):
#     pass


class ExpertMapStore():
    def __init__(
        self,
        capacity,
        num_layers,
        num_experts,
        embed_dim,
        prefetch_distance,
        device,
    ):
        self.capacity = capacity
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.prefetch_distance = prefetch_distance
        self.device = torch.device(device)
        self.dtype = torch.float32

        self.store_embed = torch.zeros(
            (capacity, embed_dim), dtype=self.dtype, device=self.device)
        self.store_traj = torch.zeros(
            (capacity, num_layers, num_experts), dtype=self.dtype, device=self.device)

        self.data_size = 0

    def import_store_data(self, state_path):
        self.store_embed = torch.from_numpy(
            np.load(f"{state_path}~embed~{self.capacity}.npy",
                    allow_pickle=False)
        ).to(self.device, dtype=self.dtype, non_blocking=True)

        self.store_traj = torch.from_numpy(
            np.load(f"{state_path}~traj~{self.capacity}.npy",
                    allow_pickle=False)
        ).to(self.device, dtype=self.dtype, non_blocking=True)

        self.data_size = self.store_embed.shape[0]

    def export_store_data(self, state_path):
        np.save(f"{state_path}~embed~{self.capacity}.npy",
                self.store_embed.detach().cpu().numpy(), allow_pickle=False)
        np.save(f"{state_path}~traj~{self.capacity}.npy",
                self.store_traj.detach().cpu().numpy(), allow_pickle=False)

    @torch.inference_mode()
    def _cosine_sim(self, A: torch.Tensor, B: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        A = torch.nn.functional.normalize(A, dim=-1, eps=eps)
        B = torch.nn.functional.normalize(B, dim=-1, eps=eps)
        return A @ B.T

    def _ensure_tensor(self, x, shape_last=None):
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.as_tensor(x)
        if shape_last is not None:
            assert t.shape[-len(shape_last):] == tuple(
                shape_last), f"Expected trailing shape {shape_last}, got {tuple(t.shape)}"
        return t.to(self.device, dtype=self.dtype, non_blocking=True)

    @torch.inference_mode()
    def add(self, embeds, expert_maps):
        embeds = self._ensure_tensor(embeds, shape_last=(self.embed_dim,))
        expert_maps = self._ensure_tensor(
            expert_maps, shape_last=(self.num_layers, self.num_experts))
        B = embeds.size(0)
        if B == 0:
            return

        free = self.capacity - self.data_size
        take = min(free, B)
        if take > 0:
            idx = slice(self.data_size, self.data_size + take)
            self.store_embed[idx] = embeds[:take]
            self.store_traj[idx] = expert_maps[:take]
            self.data_size += take

        rem = B - take
        if rem > 0:
            S_e = F.cosine_similarity(
                embeds[-rem:].unsqueeze(1),
                self.store_embed[:self.data_size].unsqueeze(0),
                dim=-1,
            )
            sims_t = F.cosine_similarity(
                expert_maps[-rem:].reshape(rem, -1).unsqueeze(1),
                self.store_traj[:self.data_size].reshape(
                    self.data_size, -1).unsqueeze(0),
                dim=-1,
            )
            S_t = sims_t
            w = self.prefetch_distance / float(self.num_layers)
            redundant = w * S_e + (1.0 - w) * S_t
            evict_idx = torch.argmax(redundant, dim=1)
            self.store_embed[evict_idx] = embeds[-rem:]
            self.store_traj[evict_idx] = expert_maps[-rem:]

        self.data_size = min(self.capacity, self.data_size)

    @torch.inference_mode()
    def match_embed(self, embeds):
        if self.data_size == 0:
            return None, None
        embeds = self._ensure_tensor(embeds, shape_last=(self.embed_dim,))
        sims = F.cosine_similarity(
            embeds.unsqueeze(1),
            self.store_embed[:self.data_size].unsqueeze(0),
            dim=-1,
        )
        scores, argmax = sims.max(dim=1)
        maps = self.store_traj[:self.data_size][argmax]
        return scores, maps

    @torch.inference_mode()
    def match_traj(self, trajs):
        if self.data_size == 0:
            return None, None
        trajs = self._ensure_tensor(trajs, shape_last=(
            trajs.shape[-2], self.num_experts))
        L_obs = trajs.shape[1]
        B = trajs.shape[0]
        sims = F.cosine_similarity(
            trajs[:, None, :L_obs, :].reshape(B, 1, -1),
            self.store_traj[:self.data_size, :L_obs, :][None,
                                                        :, :, :].reshape(1, self.data_size, -1),
            dim=-1,
        )
        scores, argmax = sims.max(dim=1)
        maps = self.store_traj[:self.data_size][argmax]
        return scores, maps


class ExpertMapMatcher():
    def __init__(
        self,
        expert_tracer,
        expert_map_store,
        expert_prefetcher,
        prefetch_distance,
    ):
        self.expert_tracer = expert_tracer
        self.expert_map_store = expert_map_store
        self.expert_prefetcher = expert_prefetcher

        self.prefetch_distance = prefetch_distance
        self.num_layers = self.expert_map_store.num_layers
        self.num_experts = self.expert_map_store.num_experts
        self.embed_dim = self.expert_map_store.embed_dim
        self.top_k = self.expert_tracer.top_k

        self.device = self.expert_map_store.device

    @torch.inference_mode()
    def _select_by_cumsum(self, probs: torch.Tensor, threshold: torch.Tensor, top_k: int):
        vals, idx = torch.sort(probs, dim=-1, descending=True)
        csum = vals.cumsum(-1)
        th = threshold.view(-1, 1) if threshold.ndim == 1 else threshold
        has_pos = vals.gt(0).any(-1)
        k = (csum <= th).sum(-1)
        k_nonzero = torch.clamp(k, min=top_k, max=vals.size(-1))
        k = torch.where(has_pos, k_nonzero, torch.zeros_like(k))
        ar = torch.arange(vals.size(-1), device=probs.device).unsqueeze(0)
        keep_sorted = ar < k.unsqueeze(-1)
        mask = torch.zeros_like(keep_sorted, dtype=torch.bool)
        mask.scatter_(dim=-1, index=idx, src=keep_sorted)
        return probs * mask

    @torch.inference_mode()
    def _layer_decay_weights(self, layer_start: int, layer_end: int) -> torch.Tensor:
        assert layer_end > layer_start
        w = torch.ones(self.num_layers, device=self.device)
        rng = torch.arange(self.num_layers, device=self.device)
        band = (rng >= layer_start) & (rng < layer_end)
        w[band] = -1 / (layer_end + 1) * (rng[band] - layer_start) + 1
        return w

    @torch.inference_mode()
    def process_expert_map(self, layer_start: int, layer_end: int,
                           score: torch.Tensor, expert_map: torch.Tensor):
        probs = expert_map.clone()
        expert_prob_map = probs
        if layer_end <= layer_start:
            return torch.zeros_like(probs), expert_prob_map
        if layer_start > 0:
            probs[:layer_start, :].zero_()
        if layer_end < self.num_layers:
            probs[layer_end:, :].zero_()
        prefetch_priority_map = self._select_by_cumsum(
            probs, torch.clamp(1 - score, 0, 1), self.top_k)
        decay = self._layer_decay_weights(
            layer_start, layer_end).unsqueeze(-1)
        prefetch_priority_map = prefetch_priority_map * decay
        prefetch_priority_map[layer_start:layer_end] += 1e-6
        return prefetch_priority_map, expert_prob_map

    @torch.inference_mode()
    def embed_prefetch(self, seq_id: int, input_embeds: torch.Tensor):
        if self.prefetch_distance <= 0:
            return
        seq_len = input_embeds.shape[0]
        scores, maps = self.expert_map_store.match_embed(input_embeds)
        if scores is not None and maps is not None:
            layer_start = 0
            layer_end = self.prefetch_distance

            for i, (s, m) in enumerate(zip(scores, maps)):
                pred_map, prob_map = self.process_expert_map(
                    layer_start, layer_end, s, m)
                iter_id = i if seq_len > 1 else -1
                self.expert_tracer.update_preds(
                    seq_id, iter_id, pred_map, layer_start, layer_end)
                self.expert_prefetcher.prefetch_experts(
                    pred_map, prob_map)

    @torch.inference_mode()
    def traj_prefetch(self, seq_id: int, input_trajs: torch.Tensor):
        if self.prefetch_distance <= 0:
            return
        seq_len = input_trajs.shape[0]
        num_layers_obs = input_trajs.shape[1]
        layer_start = num_layers_obs + self.prefetch_distance
        if layer_start < self.num_layers:
            layer_end = self.num_layers
            scores, maps = self.expert_map_store.match_traj(
                input_trajs)

            if scores is not None and maps is not None:
                for i, (s, m) in enumerate(zip(scores, maps)):
                    pred_map, prob_map = self.process_expert_map(
                        layer_start, layer_end, s, m)
                    iter_id = i if seq_len > 1 else -1
                    self.expert_tracer.update_preds(
                        seq_id, iter_id, pred_map, layer_start, layer_end)
                    self.expert_prefetcher.prefetch_experts(
                        pred_map, prob_map)


@dataclass
class ResidentRegistry:
    """Runtime-visible resident admission state.

    This keeps resident backbone metadata as a first-class runtime object
    instead of treating the resident JSON as write-only input.
    """

    enabled: bool = False
    source_file: str = ""
    selection_rule: str = ""
    layout: str = ""
    requested_expert_ids: list = field(default_factory=list)
    admitted_expert_ids: list = field(default_factory=list)
    requested_node_ids: list = field(default_factory=list)
    admitted_node_ids: list = field(default_factory=list)
    requested_count: int = 0
    admitted_count: int = 0
    requested_node_count: int = 0
    admitted_node_count: int = 0
    requested_tensor_count: int = 0
    admitted_tensor_count: int = 0
    requested_bytes: int = 0
    admitted_bytes: int = 0
    budget_bytes: int = 0
    budget_source: str = ""
    fast_path_modules: int = 0
    fast_path_expert_count: int = 0
    clipped: bool = False

    def to_public_dict(self):
        return {
            "enabled": bool(self.enabled),
            "source_file": self.source_file,
            "selection_rule": self.selection_rule,
            "layout": self.layout,
            "requested_count": int(self.requested_count),
            "admitted_count": int(self.admitted_count),
            "requested_node_count": int(self.requested_node_count),
            "admitted_node_count": int(self.admitted_node_count),
            "requested_tensor_count": int(self.requested_tensor_count),
            "admitted_tensor_count": int(self.admitted_tensor_count),
            "requested_bytes": int(self.requested_bytes),
            "admitted_bytes": int(self.admitted_bytes),
            "budget_bytes": int(self.budget_bytes),
            "budget_source": self.budget_source,
            "fast_path_modules": int(self.fast_path_modules),
            "fast_path_expert_count": int(self.fast_path_expert_count),
            "clipped": bool(self.clipped),
        }


@dataclass
class RuntimeProfile:
    """Coarse real-machine runtime attribution counters.

    The fields below are intentionally reported as wall-time rather than
    claiming exact DMA or kernel time. This keeps the payload honest while
    still exposing where the current runtime spends host-visible time.
    """

    module_begin_calls: int = 0
    module_end_calls: int = 0
    param_begin_calls: int = 0
    buffer_begin_calls: int = 0
    param_end_calls: int = 0
    buffer_end_calls: int = 0
    module_begin_wall_time_sec: float = 0.0
    module_end_wall_time_sec: float = 0.0
    resident_fastpath_module_skips: int = 0
    manual_service_module_skips: int = 0
    manual_subtree_begin_calls: int = 0
    manual_subtree_end_calls: int = 0
    manual_subtree_begin_wall_time_sec: float = 0.0
    manual_subtree_end_wall_time_sec: float = 0.0
    tail_group_begin_calls: int = 0
    tail_group_end_calls: int = 0
    tail_group_module_count: int = 0
    tail_group_expert_blocks: int = 0
    tail_group_token_assignments: int = 0
    tail_group_begin_wall_time_sec: float = 0.0
    tail_group_end_wall_time_sec: float = 0.0
    tail_group_compute_wall_time_sec: float = 0.0

    modulelist_dispatch_calls: int = 0
    modulelist_active_expert_blocks: int = 0
    modulelist_resident_expert_blocks: int = 0
    modulelist_demand_expert_blocks: int = 0
    modulelist_token_assignments: int = 0
    modulelist_resident_token_assignments: int = 0
    modulelist_demand_token_assignments: int = 0
    modulelist_expert_compute_wall_time_sec: float = 0.0
    modulelist_resident_compute_wall_time_sec: float = 0.0
    modulelist_demand_compute_wall_time_sec: float = 0.0
    resident_lane_expert_blocks: int = 0
    resident_lane_token_assignments: int = 0
    resident_lane_compute_wall_time_sec: float = 0.0

    packed_dispatch_calls: int = 0
    packed_resident_expert_blocks: int = 0
    packed_demand_expert_blocks: int = 0
    packed_resident_token_assignments: int = 0
    packed_demand_token_assignments: int = 0
    packed_resident_compute_wall_time_sec: float = 0.0
    packed_dispatch_batch_calls: int = 0
    packed_dispatch_wait_calls: int = 0
    packed_dispatch_wait_wall_time_sec: float = 0.0

    def record_module_io(
        self,
        *,
        begin_calls=0,
        end_calls=0,
        param_begin_calls=0,
        buffer_begin_calls=0,
        param_end_calls=0,
        buffer_end_calls=0,
        begin_wall_time_sec=0.0,
        end_wall_time_sec=0.0,
        skipped_fastpath=False,
        skipped_manual_service=False,
    ):
        self.module_begin_calls += int(begin_calls)
        self.module_end_calls += int(end_calls)
        self.param_begin_calls += int(param_begin_calls)
        self.buffer_begin_calls += int(buffer_begin_calls)
        self.param_end_calls += int(param_end_calls)
        self.buffer_end_calls += int(buffer_end_calls)
        self.module_begin_wall_time_sec += float(begin_wall_time_sec)
        self.module_end_wall_time_sec += float(end_wall_time_sec)
        if skipped_fastpath:
            self.resident_fastpath_module_skips += 1
        if skipped_manual_service:
            self.manual_service_module_skips += 1

    def record_manual_subtree_service(
        self,
        *,
        begin_calls=0,
        end_calls=0,
        begin_wall_time_sec=0.0,
        end_wall_time_sec=0.0,
    ):
        self.manual_subtree_begin_calls += int(begin_calls)
        self.manual_subtree_end_calls += int(end_calls)
        self.manual_subtree_begin_wall_time_sec += float(begin_wall_time_sec)
        self.manual_subtree_end_wall_time_sec += float(end_wall_time_sec)

    def record_tail_group_service(
        self,
        *,
        begin_calls=0,
        end_calls=0,
        module_count=0,
        expert_blocks=0,
        token_assignments=0,
        begin_wall_time_sec=0.0,
        end_wall_time_sec=0.0,
        compute_wall_time_sec=0.0,
    ):
        self.tail_group_begin_calls += int(begin_calls)
        self.tail_group_end_calls += int(end_calls)
        self.tail_group_module_count += int(module_count)
        self.tail_group_expert_blocks += int(expert_blocks)
        self.tail_group_token_assignments += int(token_assignments)
        self.tail_group_begin_wall_time_sec += float(begin_wall_time_sec)
        self.tail_group_end_wall_time_sec += float(end_wall_time_sec)
        self.tail_group_compute_wall_time_sec += float(compute_wall_time_sec)

    def record_modulelist_dispatch(
        self,
        *,
        active_expert_blocks,
        resident_expert_blocks,
        demand_expert_blocks,
        token_assignments,
        resident_token_assignments,
        demand_token_assignments,
        expert_compute_wall_time_sec,
        resident_compute_wall_time_sec=0.0,
        demand_compute_wall_time_sec=0.0,
    ):
        self.modulelist_dispatch_calls += 1
        self.modulelist_active_expert_blocks += int(active_expert_blocks)
        self.modulelist_resident_expert_blocks += int(resident_expert_blocks)
        self.modulelist_demand_expert_blocks += int(demand_expert_blocks)
        self.modulelist_token_assignments += int(token_assignments)
        self.modulelist_resident_token_assignments += int(resident_token_assignments)
        self.modulelist_demand_token_assignments += int(demand_token_assignments)
        self.modulelist_expert_compute_wall_time_sec += float(expert_compute_wall_time_sec)
        self.modulelist_resident_compute_wall_time_sec += float(resident_compute_wall_time_sec)
        self.modulelist_demand_compute_wall_time_sec += float(demand_compute_wall_time_sec)
        self.resident_lane_expert_blocks += int(resident_expert_blocks)
        self.resident_lane_token_assignments += int(resident_token_assignments)
        self.resident_lane_compute_wall_time_sec += float(resident_compute_wall_time_sec)

    def record_packed_dispatch(
        self,
        *,
        resident_expert_blocks,
        demand_expert_blocks,
        resident_token_assignments,
        demand_token_assignments,
        resident_compute_wall_time_sec,
        dispatch_batch_calls=0,
        dispatch_wait_calls,
        dispatch_wait_wall_time_sec,
    ):
        self.packed_dispatch_calls += 1
        self.packed_resident_expert_blocks += int(resident_expert_blocks)
        self.packed_demand_expert_blocks += int(demand_expert_blocks)
        self.packed_resident_token_assignments += int(resident_token_assignments)
        self.packed_demand_token_assignments += int(demand_token_assignments)
        self.packed_resident_compute_wall_time_sec += float(resident_compute_wall_time_sec)
        self.packed_dispatch_batch_calls += int(dispatch_batch_calls)
        self.packed_dispatch_wait_calls += int(dispatch_wait_calls)
        self.packed_dispatch_wait_wall_time_sec += float(dispatch_wait_wall_time_sec)

    def to_public_dict(self):
        return {
            "module_begin_calls": int(self.module_begin_calls),
            "module_end_calls": int(self.module_end_calls),
            "param_begin_calls": int(self.param_begin_calls),
            "buffer_begin_calls": int(self.buffer_begin_calls),
            "param_end_calls": int(self.param_end_calls),
            "buffer_end_calls": int(self.buffer_end_calls),
            "module_begin_wall_time_sec": float(self.module_begin_wall_time_sec),
            "module_end_wall_time_sec": float(self.module_end_wall_time_sec),
            "resident_fastpath_module_skips": int(self.resident_fastpath_module_skips),
            "manual_service_module_skips": int(self.manual_service_module_skips),
            "manual_subtree_begin_calls": int(self.manual_subtree_begin_calls),
            "manual_subtree_end_calls": int(self.manual_subtree_end_calls),
            "manual_subtree_begin_wall_time_sec": float(self.manual_subtree_begin_wall_time_sec),
            "manual_subtree_end_wall_time_sec": float(self.manual_subtree_end_wall_time_sec),
            "tail_group_begin_calls": int(self.tail_group_begin_calls),
            "tail_group_end_calls": int(self.tail_group_end_calls),
            "tail_group_module_count": int(self.tail_group_module_count),
            "tail_group_expert_blocks": int(self.tail_group_expert_blocks),
            "tail_group_token_assignments": int(self.tail_group_token_assignments),
            "tail_group_begin_wall_time_sec": float(self.tail_group_begin_wall_time_sec),
            "tail_group_end_wall_time_sec": float(self.tail_group_end_wall_time_sec),
            "tail_group_compute_wall_time_sec": float(self.tail_group_compute_wall_time_sec),
            "modulelist_dispatch_calls": int(self.modulelist_dispatch_calls),
            "modulelist_active_expert_blocks": int(self.modulelist_active_expert_blocks),
            "modulelist_resident_expert_blocks": int(self.modulelist_resident_expert_blocks),
            "modulelist_demand_expert_blocks": int(self.modulelist_demand_expert_blocks),
            "modulelist_token_assignments": int(self.modulelist_token_assignments),
            "modulelist_resident_token_assignments": int(self.modulelist_resident_token_assignments),
            "modulelist_demand_token_assignments": int(self.modulelist_demand_token_assignments),
            "modulelist_expert_compute_wall_time_sec": float(self.modulelist_expert_compute_wall_time_sec),
            "modulelist_resident_compute_wall_time_sec": float(self.modulelist_resident_compute_wall_time_sec),
            "modulelist_demand_compute_wall_time_sec": float(self.modulelist_demand_compute_wall_time_sec),
            "resident_lane_expert_blocks": int(self.resident_lane_expert_blocks),
            "resident_lane_token_assignments": int(self.resident_lane_token_assignments),
            "resident_lane_compute_wall_time_sec": float(self.resident_lane_compute_wall_time_sec),
            "packed_dispatch_calls": int(self.packed_dispatch_calls),
            "packed_resident_expert_blocks": int(self.packed_resident_expert_blocks),
            "packed_demand_expert_blocks": int(self.packed_demand_expert_blocks),
            "packed_resident_token_assignments": int(self.packed_resident_token_assignments),
            "packed_demand_token_assignments": int(self.packed_demand_token_assignments),
            "packed_resident_compute_wall_time_sec": float(self.packed_resident_compute_wall_time_sec),
            "packed_dispatch_batch_calls": int(self.packed_dispatch_batch_calls),
            "packed_dispatch_wait_calls": int(self.packed_dispatch_wait_calls),
            "packed_dispatch_wait_wall_time_sec": float(self.packed_dispatch_wait_wall_time_sec),
        }


@dataclass(frozen=True)
class ModuleServiceGroupContext:
    modules: tuple
    begun_by_module: tuple
    expert_blocks: int = 0
    token_assignments: int = 0


class OffloadEngine(object):
    param_id = 0
    request_id = 0
    # request_id_flag = False

    def __init__(
        self,
        capacity,
        config: PretrainedConfig,
        prefetch_distance,
        device,
        eval_mode,
    ):
        self.offload_exemption = set()
        self.expert_modules = []

        self.model_create_counter = None

        self.ckpt_files = []
        self.config = config

        if capacity is None:
            capacity = 1000

        self.num_layers, self.num_experts, self.num_encoder_layers, self.embed_dim, self.top_k = parse_moe_param(
            self.config)

        self.expert_map_store = ExpertMapStore(
            capacity=capacity,
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            embed_dim=self.embed_dim,
            prefetch_distance=prefetch_distance,
            device=device,
        )

        self.expert_tracer = ExpertTracer(
            capacity, config, self.expert_map_store, eval_mode, device)

        self.quant_method = None
        self.packed_uses_synthetic_slices = False

        self.prefetch_distance = prefetch_distance
        self.device = torch.device(device)
        self.eval_mode = eval_mode

        self.moe_layers = []
        self.resident_expert_ids = []
        self.resident_expert_ids_set = set()
        self._runtime_budget_override_bytes = None
        self._runtime_budget_override_source = ""
        self._resident_budget_override_bytes = None
        self._resident_budget_override_source = ""
        self._reset_resident_registry()
        self.runtime_profile = RuntimeProfile()

    def init_expert_map_matcher(self):
        self.expert_map_matcher = ExpertMapMatcher(
            expert_tracer=self.expert_tracer,
            expert_map_store=self.expert_map_store,
            expert_prefetcher=self.expert_prefetcher,
            prefetch_distance=self.prefetch_distance,
        )

    def _is_packed_layout(self):
        return parse_expert_layout(self.config) == "packed"

    @staticmethod
    def _is_packed_expert_container(module):
        return module.__class__.__name__ in {
            "MixtralExperts",
            "DeepseekV2Experts",
            "DeepseekV3NaiveMoe",
        }

    @staticmethod
    def _packed_role_order(param_name, schema):
        if f".{schema.gate_name}.weight" in param_name:
            return 0
        if f".{schema.down_name}.weight" in param_name:
            return 1
        if f".{schema.up_name}.weight" in param_name:
            return 2
        return 99

    def _build_packed_expert_groups(self):
        schema = get_packed_expert_schema(self.config)
        expert_groups = {}
        for name, tensor_id in self.name_id_map.items():
            try:
                layer_id, expert_id = parse_expert_id(name, self.config)
            except RuntimeError:
                continue
            if expert_id is None:
                continue
            expert_groups.setdefault((layer_id, expert_id), []).append((name, tensor_id))

        self.expert_tensor_groups = {}
        self.expert_tensor_map = {}
        for key, entries in expert_groups.items():
            ordered = [
                tensor_id
                for _, tensor_id in sorted(entries, key=lambda item: self._packed_role_order(item[0], schema))
            ]
            self.expert_tensor_groups[key] = ordered
            if ordered:
                self.expert_tensor_map[key] = ordered[0]

    def _sync_packed_runtime_mode(self, model):
        if not self._is_packed_layout():
            return

        model_has_packed_routed_tensors = False
        for name, _ in model.named_parameters(recurse=True):
            _, expert_group, _ = parse_packed_expert_tensor(name, self.config)
            if expert_group == "routed_experts":
                model_has_packed_routed_tensors = True
                break

        self.packed_uses_synthetic_slices = model_has_packed_routed_tensors
        if self.packed_uses_synthetic_slices and not self.expert_tensor_groups:
            raise RuntimeError(
                "Packed runtime detected routed packed tensors in the model, "
                "but the offload index does not contain synthetic expert slices. "
                "Remove the existing offload_path and rebuild it with the current runtime."
            )

    def _get_packed_expert_topology(self):
        if not self.packed_uses_synthetic_slices:
            return []
        expert_layers = {}
        for (layer_id, expert_id), tensor_ids in self.expert_tensor_groups.items():
            expert_layers.setdefault(layer_id, {})[expert_id] = tensor_ids

        topology = []
        for layer_id in sorted(expert_layers):
            expert_entries = expert_layers[layer_id]
            stage = []
            for expert_id in sorted(expert_entries):
                stage.append(expert_entries[expert_id])
            topology.append((f"layers.{layer_id}.mlp.experts", stage))
        return topology

    def _load_resident_expert_ids(self):
        resident_file = getattr(self.archer_config, "resident_expert_ids_file", "")
        if not resident_file:
            return []

        if not os.path.exists(resident_file):
            raise FileNotFoundError(
                f"Resident expert file does not exist: {resident_file}"
            )

        with open(resident_file, "r") as f:
            payload = json.load(f)

        selection_rule = ""
        selection_budget_bytes = None
        selection_budget_source = ""

        if isinstance(payload, dict):
            selection_rule = (
                payload.get("selection_rule")
                or payload.get("selection_method")
                or payload.get("method")
                or ""
            )
            if payload.get("selection_budget_bytes") is not None:
                selection_budget_bytes = int(payload["selection_budget_bytes"])
                selection_budget_source = payload.get("selection_budget_source", "") or ""
            if "resident_set" in payload:
                payload = payload["resident_set"]
            elif "resident_expert_ids" in payload:
                payload = payload["resident_expert_ids"]
            else:
                raise ValueError(
                    "Resident expert JSON must be a list or contain 'resident_set'."
                )

        resident_ids = []
        seen = set()
        for item in payload:
            if isinstance(item, dict):
                layer_id = int(item["layer"])
                expert_id = int(item["expert"])
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                layer_id = int(item[0])
                expert_id = int(item[1])
            else:
                raise ValueError(
                    f"Invalid resident expert entry: {item!r}"
                )
            key = (layer_id, expert_id)
            if key in seen:
                continue
            seen.add(key)
            resident_ids.append(key)

        self._record_requested_residents(
            resident_file=resident_file,
            resident_expert_ids=resident_ids,
            selection_rule=selection_rule,
            selection_budget_bytes=selection_budget_bytes,
            selection_budget_source=selection_budget_source,
        )
        return resident_ids

    def _reset_resident_registry(self):
        self.resident_registry = ResidentRegistry(
            layout=parse_expert_layout(self.config),
        )
        self._runtime_budget_override_bytes = None
        self._runtime_budget_override_source = ""
        self._resident_budget_override_bytes = None
        self._resident_budget_override_source = ""

    def _record_requested_residents(
        self,
        resident_file,
        resident_expert_ids,
        selection_rule="",
        selection_budget_bytes=None,
        selection_budget_source="",
    ):
        self.resident_registry.enabled = bool(resident_expert_ids)
        self.resident_registry.source_file = resident_file or ""
        self.resident_registry.selection_rule = selection_rule or ""
        self.resident_registry.layout = parse_expert_layout(self.config)
        self.resident_registry.requested_expert_ids = list(resident_expert_ids)
        self.resident_registry.requested_count = len(resident_expert_ids)
        self.resident_registry.requested_node_ids = []
        self.resident_registry.requested_node_count = 0
        self.resident_registry.requested_tensor_count = 0
        self.resident_registry.requested_bytes = 0
        self.resident_registry.admitted_expert_ids = []
        self.resident_registry.admitted_count = 0
        self.resident_registry.admitted_node_ids = []
        self.resident_registry.admitted_node_count = 0
        self.resident_registry.admitted_tensor_count = 0
        self.resident_registry.admitted_bytes = 0
        self.resident_registry.budget_bytes = 0
        self.resident_registry.budget_source = ""
        self.resident_registry.fast_path_modules = 0
        self.resident_registry.fast_path_expert_count = 0
        self.resident_registry.clipped = False
        self._resident_budget_override_bytes = (
            int(selection_budget_bytes)
            if selection_budget_bytes is not None
            else None
        )
        self._resident_budget_override_source = selection_budget_source or ""

    def _activate_resident_registry(self, resident_expert_ids, node_ids):
        tensor_ids = node_ids
        self.resident_expert_ids = list(resident_expert_ids)
        self.resident_expert_ids_set = set(resident_expert_ids)
        self.resident_registry.enabled = bool(resident_expert_ids)
        self.resident_registry.admitted_expert_ids = list(resident_expert_ids)
        self.resident_registry.admitted_count = len(resident_expert_ids)
        self.resident_registry.requested_count = max(
            self.resident_registry.requested_count,
            len(resident_expert_ids),
        )
        admitted_node_ids, admitted_bytes = self._collect_unique_node_stats(tensor_ids)
        self.resident_registry.admitted_node_ids = admitted_node_ids
        self.resident_registry.admitted_node_count = len(admitted_node_ids)
        self.resident_registry.admitted_tensor_count = len(tensor_ids)
        self.resident_registry.admitted_bytes = admitted_bytes
        self.resident_registry.requested_tensor_count = max(
            self.resident_registry.requested_tensor_count,
            len(tensor_ids),
        )
        self.resident_registry.requested_node_count = max(
            self.resident_registry.requested_node_count,
            len(admitted_node_ids),
        )
        self.resident_registry.requested_bytes = max(
            self.resident_registry.requested_bytes,
            admitted_bytes,
        )
        self.resident_registry.clipped = (
            self.resident_registry.requested_count
            != self.resident_registry.admitted_count
        )
        for module in getattr(self, "expert_layer_modules", []):
            layer_id = getattr(module, "layer_id", None)
            if layer_id is None:
                continue
            resident_local_ids = {
                expert_id
                for resid_layer_id, expert_id in self.resident_expert_ids_set
                if resid_layer_id == layer_id
            }
            module.resident_local_expert_ids = resident_local_ids
            module.resident_fastpath_local_expert_ids = self._resolve_resident_fastpath_ids(
                module,
                resident_local_ids,
            )
            self.resident_registry.fast_path_expert_count += len(
                module.resident_fastpath_local_expert_ids
            )

    def _collect_unique_node_stats(self, tensor_ids):
        if not tensor_ids or not hasattr(self, "archer_engine") or self.archer_engine is None:
            return [], 0
        unique_node_ids = []
        seen_node_ids = set()
        total = 0
        for tensor_id in tensor_ids:
            node_id = int(self.archer_engine.get_node_id([int(tensor_id)]))
            if node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            unique_node_ids.append(node_id)
            total += int(self.archer_engine.get_node_byte_size([int(tensor_id)]))
        return unique_node_ids, total

    def _get_sparse_budget_bytes(self):
        if not hasattr(self, "archer_engine") or self.archer_engine is None:
            return 0
        getter = getattr(self.archer_engine, "get_sparse_cache_limit", None)
        if getter is None:
            return 0
        runtime_budget_bytes = int(getter(torch.device(self.device)))
        budget_bytes = runtime_budget_bytes
        if (
            self._runtime_budget_override_bytes is not None
            and self._runtime_budget_override_bytes > 0
        ):
            if budget_bytes > 0:
                budget_bytes = min(budget_bytes, int(self._runtime_budget_override_bytes))
            else:
                budget_bytes = int(self._runtime_budget_override_bytes)
        if (
            self._resident_budget_override_bytes is not None
            and self._resident_budget_override_bytes > 0
        ):
            if budget_bytes > 0:
                return min(budget_bytes, int(self._resident_budget_override_bytes))
            return int(self._resident_budget_override_bytes)
        return budget_bytes

    def _get_sparse_budget_source(self):
        runtime_source = "free_device_memory_ratio"
        if (
            self._runtime_budget_override_bytes is not None
            and self._runtime_budget_override_bytes > 0
        ):
            if self._runtime_budget_override_source:
                runtime_source = f"min({runtime_source},{self._runtime_budget_override_source})"
            else:
                runtime_source = f"min({runtime_source},runtime_sparse_budget_override)"
        if (
            self._resident_budget_override_bytes is not None
            and self._resident_budget_override_bytes > 0
        ):
            if self._resident_budget_override_source:
                return f"min({runtime_source},{self._resident_budget_override_source})"
            return f"min({runtime_source},resident_selection_budget_bytes)"
        return runtime_source

    def get_sparse_budget_info(self):
        return {
            "budget_bytes": int(self._get_sparse_budget_bytes()),
            "budget_source": self._get_sparse_budget_source(),
        }

    def _clip_resident_prefix_to_sparse_budget(self, resident_expert_ids, expert_tensor_ids):
        budget_bytes = self._get_sparse_budget_bytes()
        self.resident_registry.budget_bytes = budget_bytes
        self.resident_registry.budget_source = self._get_sparse_budget_source()
        if budget_bytes <= 0:
            tensor_ids = [tensor_id for group in expert_tensor_ids for tensor_id in group]
            return list(resident_expert_ids), tensor_ids

        admitted_expert_ids = []
        admitted_tensor_ids = []
        seen_node_ids = set()
        admitted_bytes = 0

        for resident_key, tensor_ids in zip(resident_expert_ids, expert_tensor_ids):
            incremental_node_ids = []
            incremental_bytes = 0
            for tensor_id in tensor_ids:
                node_id = int(self.archer_engine.get_node_id([int(tensor_id)]))
                if node_id in seen_node_ids:
                    continue
                seen_node_ids.add(node_id)
                incremental_node_ids.append(node_id)
                incremental_bytes += int(self.archer_engine.get_node_byte_size([int(tensor_id)]))

            if admitted_expert_ids and admitted_bytes + incremental_bytes > budget_bytes:
                break
            if not admitted_expert_ids and incremental_bytes > budget_bytes:
                admitted_tensor_ids = []
                admitted_expert_ids = []
                break

            admitted_expert_ids.append(resident_key)
            admitted_tensor_ids.extend(tensor_ids)
            admitted_bytes += incremental_bytes

        return admitted_expert_ids, admitted_tensor_ids

    def _resolve_resident_fastpath_ids(self, module, resident_local_ids):
        if not resident_local_ids:
            return set()
        experts = getattr(module, "experts", None)
        if experts is None:
            return set()
        if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
            target_device = torch.device(self.device)
            gate_device = getattr(experts.gate_up_proj, "device", None)
            down_device = getattr(experts.down_proj, "device", None)
            if gate_device == target_device and down_device == target_device:
                return set(resident_local_ids)
            return set()
        return set()

    def get_resident_registry(self):
        return self.resident_registry.to_public_dict()

    def get_runtime_profile(self):
        return self.runtime_profile.to_public_dict()

    def _mark_module_resident_fastpath(self, module):
        """Mark a resident expert subtree to bypass generic hook bookkeeping."""
        marked = 0
        stack = [module]
        while stack:
            current = stack.pop()
            if getattr(current, "_archer_resident_fastpath", False):
                continue
            current._archer_resident_fastpath = True
            marked += 1
            stack.extend(list(current.children()))
        return marked

    def _attach_module_service_metadata(self, module):
        if getattr(module, "_archer_service_metadata_ready", False):
            return
        module._archer_service_modules = tuple(module.modules())
        module._archer_service_params = tuple(module.parameters(recurse=True))
        module._archer_service_buffers = tuple(module.buffers(recurse=True))
        module._archer_manual_service_active = False
        module._archer_service_metadata_ready = True

    def _set_manual_service_active(self, module, enabled: bool):
        self._attach_module_service_metadata(module)
        for current in module._archer_service_modules:
            current._archer_manual_service_active = enabled

    def _iter_unique_modules(self, modules):
        seen = set()
        for module in modules:
            module_id = id(module)
            if module_id in seen:
                continue
            seen.add(module_id)
            yield module

    def _set_manual_service_active_group(self, modules, enabled: bool):
        seen_service_modules = set()
        for module in self._iter_unique_modules(modules):
            self._attach_module_service_metadata(module)
            for current in module._archer_service_modules:
                current_id = id(current)
                if current_id in seen_service_modules:
                    continue
                seen_service_modules.add(current_id)
                current._archer_manual_service_active = enabled

    def _tensor_in_offload_set(self, tensor):
        return tensor.data_ptr() in self.offload_set

    def _move_tensor_to_service_device(self, tensor):
        tensor_device = tensor.device
        target_device = torch.device(self.device)
        if tensor_device != target_device:
            tensor.data = tensor.data.to(target_device)

    def _begin_manual_service_tensor(self, tensor):
        if not self._tensor_in_offload_set(tensor):
            self._move_tensor_to_service_device(tensor)
            return False
        tensor_ptr = tensor.data_ptr()
        self.offload_set.remove(tensor_ptr)
        try:
            self.archer_engine.begin(self.request_id, tensor)
            return True
        finally:
            self.offload_set.add(tensor.data_ptr())

    def _end_manual_service_tensor(self, tensor):
        if not self._tensor_in_offload_set(tensor):
            return False
        tensor_ptr = tensor.data_ptr()
        self.offload_set.remove(tensor_ptr)
        try:
            self.archer_engine.end(self.request_id, tensor)
            return True
        finally:
            self.offload_set.add(tensor.data_ptr())

    def _begin_manual_service_tensors_group(self, tensors):
        tensor_list = tuple(tensors)
        if not tensor_list:
            return ()

        removed_tensors = []
        try:
            for tensor in tensor_list:
                tensor_ptr = tensor.data_ptr()
                self.offload_set.remove(tensor_ptr)
                removed_tensors.append(tensor)

            begin_group = getattr(self.archer_engine, "begin_group", None)
            if begin_group is not None:
                begin_group(self.request_id, list(tensor_list))
            else:
                begun = []
                try:
                    for tensor in tensor_list:
                        self.archer_engine.begin(self.request_id, tensor)
                        begun.append(tensor)
                except Exception:
                    for tensor in reversed(begun):
                        self.archer_engine.end(self.request_id, tensor)
                    raise
            return tensor_list
        finally:
            for tensor in removed_tensors:
                self.offload_set.add(tensor.data_ptr())

    def _end_manual_service_tensors_group(self, tensors):
        tensor_list = tuple(tensor for tensor in tensors if self._tensor_in_offload_set(tensor))
        if not tensor_list:
            return ()

        removed_tensors = []
        try:
            for tensor in tensor_list:
                tensor_ptr = tensor.data_ptr()
                self.offload_set.remove(tensor_ptr)
                removed_tensors.append(tensor)

            end_group = getattr(self.archer_engine, "end_group", None)
            if end_group is not None:
                end_group(self.request_id, list(tensor_list))
            else:
                for tensor in tensor_list:
                    self.archer_engine.end(self.request_id, tensor)
            return tensor_list
        finally:
            for tensor in removed_tensors:
                self.offload_set.add(tensor.data_ptr())

    def _begin_module_subtree(self, module):
        self._attach_module_service_metadata(module)
        t0 = time.perf_counter()
        begun_tensors = []
        try:
            for param in module._archer_service_params:
                if self._begin_manual_service_tensor(param):
                    begun_tensors.append(param)
            for buf in module._archer_service_buffers:
                if self._begin_manual_service_tensor(buf):
                    begun_tensors.append(buf)
        except Exception:
            for tensor in reversed(begun_tensors):
                self._end_manual_service_tensor(tensor)
            raise
        self.runtime_profile.record_manual_subtree_service(
            begin_calls=1,
            begin_wall_time_sec=time.perf_counter() - t0,
        )
        return tuple(begun_tensors)

    def _begin_module_subtrees_group(self, modules):
        unique_modules = tuple(self._iter_unique_modules(modules))
        begun_by_module = []
        flat_begun_tensors = []
        for module in unique_modules:
            self._attach_module_service_metadata(module)
            module_begun_tensors = []
            for param in module._archer_service_params:
                if self._tensor_in_offload_set(param):
                    module_begun_tensors.append(param)
                else:
                    self._move_tensor_to_service_device(param)
            for buf in module._archer_service_buffers:
                if self._tensor_in_offload_set(buf):
                    module_begun_tensors.append(buf)
                else:
                    self._move_tensor_to_service_device(buf)
            begun_tensors = tuple(module_begun_tensors)
            flat_begun_tensors.extend(begun_tensors)
            begun_by_module.append((module, begun_tensors))
        if flat_begun_tensors:
            self._begin_manual_service_tensors_group(tuple(flat_begun_tensors))
        return tuple(begun_by_module)

    def _end_module_subtree(self, module, begun_tensors=None):
        self._attach_module_service_metadata(module)
        t0 = time.perf_counter()
        tensors = begun_tensors
        if tensors is None:
            tensors = (*module._archer_service_params, *module._archer_service_buffers)
        for tensor in tensors:
            self._end_manual_service_tensor(tensor)
        self.runtime_profile.record_manual_subtree_service(
            end_calls=1,
            end_wall_time_sec=time.perf_counter() - t0,
        )

    def _end_module_subtrees_group(self, begun_by_module):
        flat_begun_tensors = []
        for _, begun_tensors in reversed(tuple(begun_by_module)):
            flat_begun_tensors.extend(begun_tensors)
        if flat_begun_tensors:
            self._end_manual_service_tensors_group(tuple(flat_begun_tensors))

    def begin_module_group(self, modules, *, expert_blocks=0, token_assignments=0):
        module_list = tuple(modules)
        unique_module_count = len(tuple(self._iter_unique_modules(module_list)))
        if not module_list:
            return ModuleServiceGroupContext(
                modules=(),
                begun_by_module=(),
                expert_blocks=int(expert_blocks),
                token_assignments=int(token_assignments),
            )

        group_begin_t0 = time.perf_counter()
        self._set_manual_service_active_group(module_list, True)
        try:
            begun_by_module = self._begin_module_subtrees_group(module_list)
        except Exception:
            self._set_manual_service_active_group(module_list, False)
            raise
        self.runtime_profile.record_tail_group_service(
            begin_calls=1,
            module_count=unique_module_count,
            expert_blocks=int(expert_blocks),
            token_assignments=int(token_assignments),
            begin_wall_time_sec=time.perf_counter() - group_begin_t0,
        )
        return ModuleServiceGroupContext(
            modules=module_list,
            begun_by_module=begun_by_module,
            expert_blocks=int(expert_blocks),
            token_assignments=int(token_assignments),
        )

    def run_module_group(self, service_ctx, args_list, kwargs_list=None):
        if kwargs_list is None:
            kwargs_list = [None] * len(service_ctx.modules)
        t0 = time.perf_counter()
        outputs = []
        for module, args, kwargs in zip(service_ctx.modules, args_list, kwargs_list):
            call_args = args if isinstance(args, tuple) else (args,)
            call_kwargs = kwargs or {}
            outputs.append(module(*call_args, **call_kwargs))
        self.runtime_profile.record_tail_group_service(
            compute_wall_time_sec=time.perf_counter() - t0,
        )
        return outputs

    def end_module_group(self, service_ctx):
        if not service_ctx.modules:
            return
        t0 = time.perf_counter()
        try:
            self._end_module_subtrees_group(service_ctx.begun_by_module)
        finally:
            self._set_manual_service_active_group(service_ctx.modules, False)
            self.runtime_profile.record_tail_group_service(
                end_calls=1,
                end_wall_time_sec=time.perf_counter() - t0,
            )

    def run_module_demand_lane(self, module, *args, **kwargs):
        begun_tensors = None
        self._set_manual_service_active(module, True)
        try:
            begun_tensors = self._begin_module_subtree(module)
            return module(*args, **kwargs)
        finally:
            try:
                if begun_tensors is not None:
                    self._end_module_subtree(module, begun_tensors=begun_tensors)
            finally:
                self._set_manual_service_active(module, False)

    def run_module_demand_lane_group(self, modules, args_list, kwargs_list=None):
        module_list = tuple(modules)
        if not module_list:
            return []
        if kwargs_list is None:
            kwargs_list = [None] * len(module_list)
        token_assignments = 0
        for args in args_list:
            first_arg = args[0] if isinstance(args, tuple) else args
            if hasattr(first_arg, "shape") and len(first_arg.shape) > 0:
                token_assignments += int(first_arg.shape[0])
        service_ctx = self.begin_module_group(
            module_list,
            expert_blocks=len(module_list),
            token_assignments=token_assignments,
        )
        try:
            return self.run_module_group(service_ctx, args_list, kwargs_list=kwargs_list)
        finally:
            self.end_module_group(service_ctx)

    def pin_resident_experts(self, model, resident_expert_ids):
        """Pin backbone experts via runtime-native C++ API.
        通过 C++ 原生 API 固定 backbone expert。

        The C++ engine marks each node as is_resident=true, loads it to
        GPU, and exempts it from the LFU eviction path.  On the Python
        side we only need to remove the corresponding parameters from
        offload_set so that the forward hooks skip begin()/end().
        """
        if not resident_expert_ids:
            return

        # Collect tensor_ids (C++ node ids) for all resident experts
        # 收集所有 resident expert 的 tensor_id（C++ 节点 id）
        node_ids = []
        if self._is_packed_layout():
            expert_tensor_ids = []
            for layer_id, expert_id in resident_expert_ids:
                tensor_ids = self.expert_tensor_groups.get((layer_id, expert_id))
                if tensor_ids is None:
                    raise KeyError(
                        f"Could not find packed tensor ids for resident expert ({layer_id}, {expert_id})"
                    )
                expert_tensor_ids.append(list(tensor_ids))
                node_ids.extend(tensor_ids)
            requested_node_ids, requested_bytes = self._collect_unique_node_stats(node_ids)
            self.resident_registry.requested_node_ids = requested_node_ids
            self.resident_registry.requested_node_count = len(requested_node_ids)
            self.resident_registry.requested_tensor_count = len(node_ids)
            self.resident_registry.requested_bytes = requested_bytes
            admitted_expert_ids, admitted_tensor_ids = self._clip_resident_prefix_to_sparse_budget(
                resident_expert_ids,
                expert_tensor_ids,
            )
            if admitted_tensor_ids:
                self.archer_engine.pin_resident_nodes(admitted_tensor_ids)
            self._activate_resident_registry(admitted_expert_ids, admitted_tensor_ids)
            print(
                f"Pinned {self.resident_registry.admitted_count} resident experts "
                f"({self.resident_registry.admitted_tensor_count} tensors) on {self.device}",
                flush=True,
            )
            return

        expert_tensor_ids = []
        for layer_id, expert_id in resident_expert_ids:
            if not (0 <= layer_id < len(self.moe_layers)):
                raise IndexError(
                    f"Resident layer_id out of range: {layer_id} "
                    f"(num_moe_layers={len(self.moe_layers)})"
                )
            expert_block = self.moe_layers[layer_id]
            if not hasattr(expert_block, "experts") or not (
                0 <= expert_id < len(expert_block.experts)
            ):
                raise IndexError(
                    f"Resident expert_id out of range: "
                    f"layer={layer_id}, expert={expert_id}"
                )
            tensor_id = self.expert_tensor_map.get((layer_id, expert_id))
            if tensor_id is None:
                raise KeyError(
                    f"Could not find tensor id for resident expert "
                    f"({layer_id}, {expert_id})"
                )
            expert_tensor_ids.append([tensor_id])
            node_ids.append(tensor_id)
        requested_node_ids, requested_bytes = self._collect_unique_node_stats(node_ids)
        self.resident_registry.requested_node_ids = requested_node_ids
        self.resident_registry.requested_node_count = len(requested_node_ids)
        self.resident_registry.requested_tensor_count = len(node_ids)
        self.resident_registry.requested_bytes = requested_bytes
        admitted_resident_ids, admitted_node_ids = self._clip_resident_prefix_to_sparse_budget(
            resident_expert_ids,
            expert_tensor_ids,
        )

        # C++ native pin: load to GPU + mark is_resident + exempt from eviction
        # C++ 原生 pin：加载到 GPU + 标记 is_resident + 驱逐豁免
        if admitted_node_ids:
            self.archer_engine.pin_resident_nodes(admitted_node_ids)

        # Materialize Python tensor data pointers via begin().
        # The C++ engine loaded data to GPU internally, but Python tensors
        # still point to placeholder memory.  begin() updates them.
        # is_resident=true guarantees no eviction, so begin() is safe here.
        # 通过 begin() 更新 Python tensor 的 data pointer。
        # C++ 引擎已经把数据加载到 GPU，但 Python tensor 仍指向占位内存。
        # begin() 会更新它们。is_resident=true 保证不被驱逐，所以这里安全。
        pinned_tensors = 0
        fast_path_modules = 0
        for layer_id, expert_id in admitted_resident_ids:
            expert_module = self.moe_layers[layer_id].experts[expert_id]
            for param in expert_module.parameters(recurse=True):
                if param.data.data_ptr() in self.offload_set:
                    self.offload_set.remove(param.data.data_ptr())
                    self.archer_engine.begin(self.request_id, param)
                    # begin() may change data_ptr; track the new one
                    # begin() 可能改变 data_ptr，记录新的
                    self.offload_set.add(param.data.data_ptr())
                # Now remove permanently from offload_set (no end() ever)
                # 从 offload_set 永久移除（不再调 end()）
                self.offload_set.discard(param.data.data_ptr())
                pinned_tensors += 1
            for buf in expert_module.buffers(recurse=True):
                if buf.data_ptr() in self.offload_set:
                    self.offload_set.remove(buf.data_ptr())
                    self.archer_engine.begin(self.request_id, buf)
                    self.offload_set.add(buf.data_ptr())
                self.offload_set.discard(buf.data_ptr())
                pinned_tensors += 1
            fast_path_modules += self._mark_module_resident_fastpath(expert_module)

        self._activate_resident_registry(admitted_resident_ids, admitted_node_ids)
        self.resident_registry.admitted_tensor_count = pinned_tensors
        self.resident_registry.fast_path_modules = fast_path_modules
        print(
            f"Pinned {self.resident_registry.admitted_count} resident experts "
            f"({pinned_tensors} tensors) on {self.device}",
            flush=True,
        )

    def init(
        self, cls: Type[PreTrainedModel], ar_config: Union[str, Dict, ArcherConfig]
    ):

        self.cls = cls
        self.name_id_map = {}
        self.synthetic_name_map = {}
        self.tensor_id_map = {}
        self.registered_tensors = set()
        self.forward_hooks = []
        self.backward_hooks = []

        self.offload_set = set()

        if isinstance(ar_config, str):
            _archer_config = ArcherConfig.load_from_file(ar_config)
        elif isinstance(ar_config, dict):
            _archer_config = ArcherConfig.load_from_json(ar_config)
        elif isinstance(ar_config, ArcherConfig):
            _archer_config = ar_config
        else:
            raise ValueError(
                "ArcherConfig is not provided. Please provide a path to a config file or a dict."
            )

        # TODO: get trace from trace_path

        self.checkpoint = _archer_config.offload_path

        os.makedirs(self.checkpoint, exist_ok=True)

        self.prefetch_lib = PrefetchBuilder().load() if use_jit else prefetch_op
        self.archer_engine = self.prefetch_lib.prefetch_handle(
            self.checkpoint,
            _archer_config.device_memory_ratio,
            self.device,
        )

        self.archer_config = _archer_config
        if getattr(_archer_config, "sparse_budget_bytes_override", 0):
            self._runtime_budget_override_bytes = int(_archer_config.sparse_budget_bytes_override)
            self._runtime_budget_override_source = "runtime_sparse_budget_override"
            if hasattr(self.archer_engine, "set_sparse_cache_limit"):
                self.archer_engine.set_sparse_cache_limit(
                    self.device,
                    self._runtime_budget_override_bytes,
                )

        self.expert_tracer.offload_engine = self

        return self

    def __enter__(self):

        def do_nothing_decorator(orig_func: Callable) -> Callable:

            @functools.wraps(orig_func)
            def do_nothing(*args, **kwargs):
                pass

            return do_nothing

        def post_init_decorator(orig_post_init: Callable) -> Callable:
            # FIXME: this is a hacky way to get rid of the write to weight in the post_init, need a better way to do this if we need to support model training
            @functools.wraps(orig_post_init)
            def archer_post_init(cls, *args, **kwargs):
                pass

            return archer_post_init

        def torch_index_select_decorator(orig_torch_index_select: Callable):

            @functools.wraps(orig_torch_index_select)
            def archer_torch_index_select(input, dim, index):
                return orig_torch_index_select(input, dim, index.to(input.device)).to(self.device)

            return archer_torch_index_select

        def apply_to_model_decorator(orig_apply_to_model: Callable) -> Callable:

            @functools.wraps(orig_apply_to_model)
            def archer_apply_to_model(cls, fn):
                for name, param in cls.named_parameters(recurse=True):
                    if name not in self.name_id_map:
                        continue
                    param.data = torch.zeros(
                        1, dtype=param.dtype, device=param.device, pin_memory=True
                    )

                for name, buffer in cls.named_buffers(recurse=True):
                    if name not in self.name_id_map:
                        continue
                    buffer.data = torch.zeros(
                        1, dtype=buffer.dtype, device=buffer.device, pin_memory=True
                    )

            return archer_apply_to_model

        def init_decorator(orig_init: Callable) -> Callable:

            @functools.wraps(orig_init)
            def archer_init(cls, config, *args, **kwargs):
                # self.config = config
                pass

            return archer_init

        def param_init_decorator(orig_param_init: Callable) -> Callable:

            @functools.wraps(orig_param_init)
            def archer_param_init(cls, *args, **kwargs):
                orig_param_init(cls, *args, **kwargs)

                cls.param_real_shape = {}
                for name, param in cls.named_parameters(recurse=False):
                    cls.param_real_shape[name] = param.shape
                    param.data = torch.zeros(
                        1, dtype=param.dtype, device=param.device)
                    self.model_create_counter.update(1)

                for name, buf in cls.named_buffers(recurse=False):
                    cls.param_real_shape[name] = buf.shape
                    buf.data = torch.zeros(
                        1, dtype=buf.dtype, device=buf.device)
                    self.model_create_counter.update(1)

            return archer_param_init

        def cast_classifier_decorator(orig_cast_classifier: Callable) -> Callable:

            @functools.wraps(orig_cast_classifier)
            def archer_cast_classifier(cls, *args, **kwargs):
                orig_data_ptr = cls.classifier.weight.data.data_ptr()
                if orig_data_ptr in self.offload_set:
                    self.offload_set.remove(
                        cls.classifier.weight.data.data_ptr())
                    orig_cast_classifier(cls, *args, **kwargs)
                    new_data_ptr = cls.classifier.weight.data.data_ptr()
                    self.offload_set.add(cls.classifier.weight.data.data_ptr())
                    self.archer_engine.update_tensor_map(
                        orig_data_ptr, new_data_ptr)
                else:
                    orig_cast_classifier(cls, *args, **kwargs)
                    self.offload_set.add(cls.classifier.weight.data.data_ptr())

            return archer_cast_classifier

        self.cls._old_init = self.cls.__init__
        self.cls.__init__ = init_decorator(self.cls._old_init)
        torch.nn.modules.module.Module._old_apply = torch.nn.modules.module.Module.apply
        torch.nn.modules.module.Module.apply = apply_to_model_decorator(
            torch.nn.modules.module.Module._old_apply
        )

        torch._old_index_select = torch.index_select
        torch.index_select = torch_index_select_decorator(
            torch._old_index_select)
        torch.Tensor._old_index_select = torch.Tensor.index_select
        torch.Tensor.index_select = torch_index_select_decorator(
            torch.Tensor._old_index_select
        )

        self.cls._old_post_init = self.cls.post_init
        self.cls.post_init = post_init_decorator(self.cls._old_post_init)
        PreTrainedModel._old_post_init = PreTrainedModel.post_init
        PreTrainedModel.post_init = post_init_decorator(
            PreTrainedModel._old_post_init)

        for name, module in torch.nn.modules.__dict__.items():
            if not isinstance(module, type):
                continue
            if not issubclass(module, torch.nn.modules.module.Module):
                continue
            if name in [
                "Module",
                "Sequential",
                "ModuleDict",
                "ModuleList",
                "ParameterList",
                "ParameterDict",
            ]:
                continue
            module._old_init = module.__init__
            module.__init__ = param_init_decorator(module.__init__)

            if hasattr(module, "reset_parameters"):
                module._old_reset_parameters = module.reset_parameters
                module.reset_parameters = do_nothing_decorator(
                    module.reset_parameters)

        finemoe.models.modeling_qwen.modeling_qwen2_moe._old_sparse_mlp = (
            finemoe.models.modeling_qwen.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock
        )
        finemoe.models.modeling_qwen.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock = (
            SyncQwen2MoeSparseMoeBlock
        )
        finemoe.models.modeling_olmoe.modeling_olmoe._old_sparse_mlp = (
            finemoe.models.modeling_olmoe.modeling_olmoe.OlmoeSparseMoeBlock
        )
        finemoe.models.modeling_olmoe.modeling_olmoe.OlmoeSparseMoeBlock = (
            SyncOlmoeSparseMoeBlock
        )
        if SyncMixtralSparseMoeBlock is not None:
            transformers.models.mixtral.modeling_mixtral._old_sparse_mlp = (
                transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock
            )
            transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock = (
                SyncMixtralSparseMoeBlock
            )
        if SyncDeepseekV2Moe is not None:
            transformers.models.deepseek_v2.modeling_deepseek_v2._old_sparse_mlp = (
                transformers.models.deepseek_v2.modeling_deepseek_v2.DeepseekV2Moe
            )
            transformers.models.deepseek_v2.modeling_deepseek_v2.DeepseekV2Moe = (
                SyncDeepseekV2Moe
            )
        if SyncDeepseekV3MoE is not None:
            transformers.models.deepseek_v3.modeling_deepseek_v3._old_sparse_mlp = (
                transformers.models.deepseek_v3.modeling_deepseek_v3.DeepseekV3MoE
            )
            transformers.models.deepseek_v3.modeling_deepseek_v3.DeepseekV3MoE = (
                SyncDeepseekV3MoE
            )

        def from_pretrained_decorator(orig_from_pretrained: Callable) -> Callable:

            @functools.wraps(orig_from_pretrained)
            def archer_from_pretrained(cls, *args, **kwargs):
                # print("Creating model from scratch ...")

                name_id_map_file = os.path.join(
                    self.checkpoint, "name_id_map.json")

                model_name = args[0]
                self.dtype = parse_expert_dtype(self.config)

                self.dtype_cls = self.config.torch_dtype

                if (
                    not self.archer_engine.is_tensor_index_initialized()
                    or not os.path.exists(name_id_map_file)
                ):
                    print("Creating model from scratch ...", flush=True)

                    self.cls.__init__ = self.cls._old_init

                    empty_state_dict = {}
                    self.name_id_map = {}
                    for ckpt in tqdm(
                        self.ckpt_files, desc="Loading checkpoint files", smoothing=0
                    ):
                        state_dict = {}
                        if "safetensors" in ckpt:
                            with safe_open(ckpt, framework="pt", device="cpu") as f:
                                for k in f.keys():
                                    state_dict[k] = f.get_tensor(k)
                        else:
                            state_dict = torch.load(ckpt)

                        # convert all tensors in state_dict to self.dtype
                        for k, v in state_dict.items():
                            state_dict[k] = v.to(self.dtype).to("cpu")

                        self._offload_state_dict(state_dict, empty_state_dict)

                        # print("Loading ckpt file", ckpt, flush=True)

                        del state_dict
                        gc.collect()
                        torch.cuda.empty_cache()

                    with open(name_id_map_file, "w") as f:
                        json.dump(self.name_id_map, f)
                else:
                    print("Loading model from offload_path ...", flush=True)
                    self.cls.__init__ = self.cls._old_init
                    # load the name_id_map
                    with open(name_id_map_file, "r") as f:
                        self.name_id_map = json.load(f)

                # print(self.name_id_map, flush=True)

                # get max tensor id from the name_id_map
                max_tensor_id = max(self.name_id_map.values())
                self.model_create_counter = tqdm(
                    total=max_tensor_id, desc="Model create"
                )

                is_flash_attn_available = kwargs.get(
                    "is_flash_attn_available", False)
                if self.dtype_cls is torch.bfloat16 or self.dtype_cls is torch.float16:
                    requested_attn_impl = "flash_attention_2" if is_flash_attn_available else "eager"
                    try:
                        model = cls._from_config(
                            self.config,
                            torch_dtype=self.dtype_cls,
                            attn_implementation=requested_attn_impl,
                        )
                    except ValueError as exc:
                        if requested_attn_impl != "flash_attention_2" or "Flash Attention 2" not in str(exc):
                            raise
                        print(
                            f"[WARNING] {cls.__name__} rejected flash_attention_2 on the current backend; "
                            "falling back to eager attention.",
                            flush=True,
                        )
                        model = cls._from_config(
                            self.config,
                            torch_dtype=self.dtype_cls,
                            attn_implementation="eager",
                        )
                else:
                    model = cls._from_config(
                        self.config,
                    )

                base_model_prefix = model.base_model_prefix
                # model = model.to(self.dtype).to("cpu")

                # print("Model created with dtype", self.dtype, flush=True)
                # for name, param in model.named_parameters(recurse=False):
                #     print(name, param.dtype, flush=True)

                # print(self.config, flush=True)

                if hasattr(self.config, "quantization_config"):
                    self.quant_method = self.config.quantization_config["quant_method"]
                    self.config.quantization_config["use_exllama"] = False
                    self.config.quantization_config["disable_exllama"] = True
                    # print("Quantizing model ...", self.quant_method, flush=True)
                    if self.quant_method == "gptq":
                        from optimum.gptq import GPTQQuantizer
                        # print("Quantizing model with GPTQ ...", self.config.quantization_config, flush=True)
                        optimum_quantizer = GPTQQuantizer.from_dict(
                            self.config.quantization_config
                        )

                        model = optimum_quantizer.convert_model(model)

                self.expert_prefetcher = ExpertPrefetcher(
                    self.config, self.device)
                self.expert_prefetcher.set_archer_engine(self.archer_engine)
                self.expert_dispatcher = self.prefetch_lib.expert_dispatcher(
                    self.num_experts,
                    self.num_layers,
                    parse_expert_dtype_id(self.config),
                    parse_expert_type(self.config),
                )

                for name, param in model.named_parameters(recurse=True):
                    # remove base_model_prefix from self.name_id_map
                    if name.startswith(base_model_prefix):
                        name_without_prefix = name[(
                            len(base_model_prefix) + 1):]
                        if name_without_prefix in self.name_id_map:
                            self.name_id_map[name] = self.name_id_map[
                                name_without_prefix
                            ]
                            self.name_id_map.pop(name_without_prefix)
                    param.ar_id = self.name_id_map.get(name, None)

                if not "lm_head.weight" in self.name_id_map:
                    print("lm_head.weight not in name_id_map, add it as embed_tokens")
                    self.name_id_map["lm_head.weight"] = 0
                    self.name_id_map["encoder.embed_tokens.weight"] = 0
                    self.name_id_map["decoder.embed_tokens.weight"] = 0

                    model.lm_head.weight.ar_id = 0
                    model.model.encoder.embed_tokens.weight.ar_id = 0
                    model.model.decoder.embed_tokens.weight.ar_id = 0

                if self._is_packed_layout():
                    self._build_packed_expert_groups()
                    self._sync_packed_runtime_mode(model)
                else:
                    self.expert_tensor_map = dict()
                    self.expert_tensor_groups = {}
                    for name, id in self.name_id_map.items():
                        layer_id, expert_id = parse_expert_id(name, self.config)
                        if expert_id is not None:
                            self.expert_tensor_map[(layer_id, expert_id)] = id
                            self.expert_tensor_groups[(layer_id, expert_id)] = [id]

                self.expert_prefetcher.expert_tensor_map = self.expert_tensor_map

                if self.prefetch_distance > 0:
                    self.init_expert_map_matcher()
                else:
                    self.expert_map_matcher = None

                model.expert_prefetcher = self.expert_prefetcher
                model.expert_tracer = self.expert_tracer
                model.expert_map_matcher = self.expert_map_matcher
                model._device = self.device
                model.model.expert_prefetcher = self.expert_prefetcher
                model.model.expert_tracer = self.expert_tracer
                model.model.expert_map_matcher = self.expert_map_matcher
                model.model._device = self.device

                module_idx = 0
                self.expert_layer_modules = []
                for module in model.modules():
                    sync_sparse_types = tuple(
                        t
                        for t in (
                            SyncQwen2MoeSparseMoeBlock,
                            SyncOlmoeSparseMoeBlock,
                            SyncMixtralSparseMoeBlock,
                            SyncDeepseekV2Moe,
                            SyncDeepseekV3MoE,
                        )
                        if t is not None
                    )
                    if isinstance(module, sync_sparse_types):
                        # module.archer_prefetch = self.archer_prefetch
                        # module.archer_tracer = self.archer_tracer
                        module.archer_engine = self.archer_engine
                        module.archer_config = self.archer_config
                        module.expert_dispatcher = self.expert_dispatcher
                        self.expert_modules.append(module)
                        # module.expert_executor = self.expert_executor
                        module.expert_prefetcher = self.expert_prefetcher
                        module.expert_tracer = self.expert_tracer
                        module.expert_map_matcher = self.expert_map_matcher
                        module.expert_tensor_map = self.expert_tensor_map
                        module.prefetch_distance = self.prefetch_distance
                        module.device = self.device
                        module.runtime_profile = self.runtime_profile

                        self.expert_layer_modules.append(module)

                        module.layer_id = module_idx

                        module_idx += 1

                        self.moe_layers.append(module)
                        module.moe_layers = self.moe_layers

                    if isinstance(module, (Qwen2MoeMLP, OlmoeMLP)):
                        module.offload_engine = self
                        self._attach_module_service_metadata(module)

                self.setup_archer_hooks(model)
                resident_expert_ids = self._load_resident_expert_ids()
                if resident_expert_ids:
                    self.pin_resident_experts(model, resident_expert_ids)
                    self.expert_prefetcher.set_resident_experts(
                        getattr(self, "resident_expert_ids", resident_expert_ids)
                    )
                # print("OffloadEngine init done, rank", dist.get_rank(), flush=True)
                return model

            return archer_from_pretrained

        self.cls._old_from_pretrained = self.cls.from_pretrained
        self.cls.from_pretrained = classmethod(
            from_pretrained_decorator(self.cls.from_pretrained)
        )

        return self

    # clean up initialization hooks
    def __exit__(self, exc_type, exc_value, traceback):

        # GPTQ Override
        # QuantLinear.__init__ = QuantLinear._old_init
        # QuantLinearOld.__init__ = QuantLinearOld._old_init

        self.cls.__init__ = self.cls._old_init
        self.cls.from_pretrained = self.cls._old_from_pretrained
        torch.nn.modules.module.Module.apply = torch.nn.modules.module.Module._old_apply
        torch.index_select = torch._old_index_select
        torch.Tensor.index_select = torch.Tensor._old_index_select

        self.cls.post_init = self.cls._old_post_init
        PreTrainedModel.post_init = PreTrainedModel._old_post_init

        for name, module in torch.nn.modules.__dict__.items():
            if not isinstance(module, type):
                continue
            if not issubclass(module, torch.nn.modules.module.Module):
                continue
            if name in [
                "Module",
                "Sequential",
                "ModuleDict",
                "ModuleList",
                "ParameterList",
                "ParameterDict",
            ]:
                continue
            module.__init__ = module._old_init

            if hasattr(module, "reset_parameters"):
                module.reset_parameters = module._old_reset_parameters

        finemoe.models.modeling_qwen.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock = (
            finemoe.models.modeling_qwen.modeling_qwen2_moe._old_sparse_mlp
        )
        finemoe.models.modeling_olmoe.modeling_olmoe.OlmoeSparseMoeBlock = (
            finemoe.models.modeling_olmoe.modeling_olmoe._old_sparse_mlp
        )
        mixtral_mod = getattr(getattr(transformers.models, "mixtral", None), "modeling_mixtral", None)
        if mixtral_mod is not None and hasattr(mixtral_mod, "_old_sparse_mlp"):
            mixtral_mod.MixtralSparseMoeBlock = mixtral_mod._old_sparse_mlp
        deepseek_v2_mod = getattr(getattr(transformers.models, "deepseek_v2", None), "modeling_deepseek_v2", None)
        if deepseek_v2_mod is not None and hasattr(deepseek_v2_mod, "_old_sparse_mlp"):
            deepseek_v2_mod.DeepseekV2Moe = deepseek_v2_mod._old_sparse_mlp
        deepseek_v3_mod = getattr(getattr(transformers.models, "deepseek_v3", None), "modeling_deepseek_v3", None)
        if deepseek_v3_mod is not None and hasattr(deepseek_v3_mod, "_old_sparse_mlp"):
            deepseek_v3_mod.DeepseekV3MoE = deepseek_v3_mod._old_sparse_mlp

    def get_topology(self, model):
        name_lst = []
        ret_dict = {}

        # print("Getting topology ...", self.name_id_map)

        # for name in model.state_dict().keys():
        for name, _ in model.named_parameters(recurse=True):
            match = re.search(r"\d+", name)
            if name not in self.name_id_map:
                print("param not in self.name_id_map", name)
                continue
            if match:
                if "expert" in name and "shared_expert" not in name:
                    match = re.match(r"(.*experts)", name)
                    assert match, "Not correct expert name!"
                    stored_name = match.group(1)
                    components = name.split(".")
                    # Use negative indexing to get the component between the last third and second dot
                    expert_name = components[-3]
                    if stored_name in name_lst:
                        if expert_name in ret_dict[stored_name]:
                            ret_dict[stored_name][expert_name].append(
                                self.name_id_map[name]
                            )
                        else:
                            ret_dict[stored_name][expert_name] = [
                                self.name_id_map[name]
                            ]
                    else:
                        ret_dict[stored_name] = {
                            expert_name: [self.name_id_map[name]]}
                        name_lst.append(stored_name)

                else:
                    match = re.match(r"(.*\.\d+\.)", name)
                    last_number_position = match.end() - 2
                    stored_name = name[: last_number_position + 1]

                    if stored_name in name_lst:
                        ret_dict[stored_name][0].append(self.name_id_map[name])
                    else:
                        ret_dict[stored_name] = [[self.name_id_map[name]]]
                        name_lst.append(stored_name)

            else:
                components = name.rsplit(".", 1)
                stored_name = components[0]

                if stored_name in name_lst:
                    ret_dict[stored_name][0].append(self.name_id_map[name])
                else:
                    ret_dict[stored_name] = [[self.name_id_map[name]]]
                    name_lst.append(stored_name)

        for name, _ in model.named_buffers(recurse=True):
            match = re.search(r"\d+", name)
            if name not in self.name_id_map:
                # print("buffer not in self.name_id_map", name)
                continue
            if match:
                if "expert" in name and "shared_expert" not in name:
                    match = re.match(r"(.*experts)", name)
                    assert match, "Not correct expert name!"
                    stored_name = match.group(1)
                    components = name.split(".")
                    # Use negative indexing to get the component between the last third and second dot
                    expert_name = components[-3]
                    if stored_name in name_lst:
                        if expert_name in ret_dict[stored_name]:
                            ret_dict[stored_name][expert_name].append(
                                self.name_id_map[name]
                            )
                        else:
                            ret_dict[stored_name][expert_name] = [
                                self.name_id_map[name]
                            ]
                    else:
                        ret_dict[stored_name] = {
                            expert_name: [self.name_id_map[name]]}
                        name_lst.append(stored_name)

                else:
                    matches = [match for match in re.finditer(r"\d", name)]
                    last_number_position = matches[-1].start() if matches else -1
                    stored_name = name[: last_number_position + 1]

                    if stored_name in name_lst:
                        ret_dict[stored_name][0].append(self.name_id_map[name])
                    else:
                        ret_dict[stored_name] = [[self.name_id_map[name]]]
                        name_lst.append(stored_name)
            else:
                components = name.rsplit(".", 1)
                stored_name = components[0]

                if stored_name in name_lst:
                    ret_dict[stored_name][0].append(self.name_id_map[name])
                else:
                    ret_dict[stored_name] = [[self.name_id_map[name]]]
                    name_lst.append(stored_name)

        for i in ret_dict.keys():
            if isinstance(ret_dict[i], dict):
                ret_dict[i] = list(ret_dict[i].values())

        topology = list(ret_dict.items())
        if self._is_packed_layout():
            topology.extend(self._get_packed_expert_topology())
        return topology

    def setup_archer_hooks(self, model):
        for name, param in model.named_parameters(recurse=True):
            if name not in self.name_id_map:
                continue
            self.archer_engine.register(param.data, self.name_id_map[name])
            self.offload_set.add(param.data.data_ptr())

            if "shared" in name:
                self.offload_exemption.add(param.data.data_ptr())

        for name, buffer in model.named_buffers(recurse=True):
            if name not in self.name_id_map:
                continue
            self.archer_engine.register(buffer.data, self.name_id_map[name])
            self.offload_set.add(buffer.data.data_ptr())

        topo = self.get_topology(model)
        self.archer_engine.set_topology(topo, self.device)

        @torch.no_grad()
        def _pre_forward_input_hook(module, input, kwargs, device, tensors):
            # print("pre_forward_input_hook", device, input, tensors)
            self.archer_engine.fetch_tensors(self.request_id, tensors)
            new_args = copy_args_to_device(device, input)
            new_kwargs = copy_kwargs_to_device(device, kwargs)
            return new_args, new_kwargs

        @torch.no_grad()
        def _post_forward_output_hook(module, input, output, device, tensors):
            if isinstance(output, tuple):
                new_args = copy_args_to_device(device, output)
            elif isinstance(output, dict):
                new_args = copy_kwargs_to_device(device, output)
            else:
                new_args = output.to(device)
            return new_args

        def gen_args_hook(key, input_device_index, output_device_index, tensors):

            keys = key.split(".")
            # print(keys)
            m = model
            for k in keys:
                if k.isdigit():
                    m = m[int(k)]
                else:
                    m = getattr(m, k)

            m.register_forward_pre_hook(
                functools.partial(
                    _pre_forward_input_hook, device=input_device_index, tensors=tensors
                ),
                prepend=True,
                with_kwargs=True,
            )
            if "lm_head" in key:
                m.register_forward_hook(
                    functools.partial(
                        _post_forward_output_hook, device=self.device, tensors=tensors
                    ),
                    prepend=False,
                )

        expert_layer_id = 0
        output_device_index = None
        for key, tensors in topo:
            # print(key, tensors)
            if "shared" in key or "lm_head" in key:
                key = key.split(".")[0]
                output_device_index = 0

            if "expert" in key:
                for expert_idx, expert_tensors in enumerate(tensors):
                    self.expert_dispatcher.register_expert(
                        expert_layer_id, expert_idx, expert_tensors
                    )
                    if not self._is_packed_layout():
                        if self.config.model_type in ("qwen2_moe", "olmoe"):
                            expert_key = f"{key}.{expert_idx}"
                        else:
                            expert_key = f"{key}.expert_{expert_idx}"
                        input_device_index = self.archer_engine.get_node_default_device(
                            expert_tensors
                        )
                        gen_args_hook(
                            expert_key,
                            input_device_index,
                            output_device_index,
                            expert_tensors,
                        )
                expert_layer_id += 1
            else:
                input_device_index = self.archer_engine.get_node_default_device(
                    tensors[0]
                )
                gen_args_hook(key, input_device_index,
                              output_device_index, tensors[0])
                output_device_index = input_device_index

        # @torch.no_grad()
        # def request_id_hook(module, *args):
        #     self.request_id_flag = False
        #     # self.archer_tracer.clear_request_id()
        #     # self.archer_prefetch.clear_request()

        # model.register_forward_hook(request_id_hook)

        # likely one of them should be enough but just to be safe
        self._register_hooks_recursively(model)

    def _generate_param_id(self):
        param_id = self.param_id
        self.param_id += 1
        return param_id

    def _generate_request_id(self):
        request_id = self.request_id
        self.request_id += 1
        return request_id

    def _offload_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        empty_state_dict: Dict[str, torch.Tensor],
    ) -> None:
        if parse_expert_layout(self.config) == "packed":
            already_expanded = False
            for param_name in state_dict.keys():
                try:
                    _, expert_id = parse_expert_id(param_name, self.config)
                except RuntimeError:
                    continue
                if expert_id is not None:
                    already_expanded = True
                    break

            self.packed_uses_synthetic_slices = not already_expanded
            if already_expanded:
                for param_name, tensor in state_dict.items():
                    tensor_id = self._generate_param_id()
                    self.name_id_map[param_name] = tensor_id
                    if not self.archer_engine.is_tensor_offloaded(tensor_id):
                        self.archer_engine.offload(tensor, tensor_id)
            else:
                for entry in expand_state_dict_for_offload(state_dict, self.config):
                    tensor_id = self._generate_param_id()
                    self.name_id_map[entry.name] = tensor_id
                    self.synthetic_name_map[entry.name] = entry.source_name
                    if not self.archer_engine.is_tensor_offloaded(tensor_id):
                        self.archer_engine.offload(entry.tensor, tensor_id)
        else:
            param_names = list(state_dict.keys())

            for param_name in param_names:
                self.name_id_map[param_name] = self._generate_param_id()
                if not self.archer_engine.is_tensor_offloaded(self.name_id_map[param_name]):
                    self.archer_engine.offload(
                        state_dict[param_name], self.name_id_map[param_name]
                    )

        gc.collect()
        torch.cuda.empty_cache()

    def _register_hooks_recursively(self, module, count=[0]):
        my_count = count[0]
        module.id = my_count

        if self._is_packed_layout() and self._is_packed_expert_container(module):
            return

        for child in module.children():
            count[0] = count[0] + 1
            self._register_hooks_recursively(child, count=count)

        @torch.no_grad()
        def _pre_forward_module_hook(module, args, kwargs):
            if getattr(module, "_archer_resident_fastpath", False):
                self.runtime_profile.record_module_io(skipped_fastpath=True)
                return
            if getattr(module, "_archer_manual_service_active", False):
                self.runtime_profile.record_module_io(skipped_manual_service=True)
                return
            # if self.request_id_flag == False:
            #     self.request_id_flag = True
            #     # print(kwargs, args, type(module))

            #     request_id = self._generate_request_id()
            #     # self.archer_tracer.set_request_id(request_id)
            #     # self.archer_prefetch.set_request(request_id)

            device_list = []
            param_begin_calls = 0
            buffer_begin_calls = 0
            t0 = time.perf_counter()

            for name, param in module.named_parameters(recurse=False):
                if not param.data.data_ptr() in self.offload_set:
                    param.data = param.data.to(self.device)
                    continue

                self.offload_set.remove(param.data.data_ptr())
                self.archer_engine.begin(self.request_id, param)
                self.offload_set.add(param.data.data_ptr())
                param_begin_calls += 1

                device_list.append(param.data.device)

            for name, buf in module.named_buffers(recurse=False):

                if not buf.data.data_ptr() in self.offload_set:
                    buf.data = buf.data.to(self.device)
                    continue

                # print("offload buffer", name, buf.data.data_ptr())

                self.offload_set.remove(buf.data_ptr())
                self.archer_engine.begin(self.request_id, buf)
                # buf = buf.to(self.dtype)
                self.offload_set.add(buf.data_ptr())
                buffer_begin_calls += 1

                device_list.append(buf.data.device)
            self.runtime_profile.record_module_io(
                begin_calls=1,
                param_begin_calls=param_begin_calls,
                buffer_begin_calls=buffer_begin_calls,
                begin_wall_time_sec=time.perf_counter() - t0,
            )

        @torch.no_grad()
        def _post_forward_module_hook(module, input, output):
            if getattr(module, "_archer_resident_fastpath", False):
                self.runtime_profile.record_module_io(skipped_fastpath=True)
                return
            if getattr(module, "_archer_manual_service_active", False):
                self.runtime_profile.record_module_io(skipped_manual_service=True)
                return
            device_list = []
            param_not_offload = set()
            param_end_calls = 0
            buffer_end_calls = 0
            t0 = time.perf_counter()
            for param in module.parameters(recurse=False):

                if not param.data.data_ptr() in self.offload_set:
                    param_not_offload.add(param.data.data_ptr())
                    continue

                self.offload_set.remove(param.data.data_ptr())
                self.archer_engine.end(self.request_id, param)
                self.offload_set.add(param.data.data_ptr())
                param_end_calls += 1

                device_list.append(param.data.device)

            for buf in module.buffers(recurse=False):

                if not buf.data_ptr() in self.offload_set:
                    continue

                self.offload_set.remove(buf.data_ptr())
                self.archer_engine.end(self.request_id, buf)
                self.offload_set.add(buf.data_ptr())
                buffer_end_calls += 1

                device_list.append(buf.device)
            self.runtime_profile.record_module_io(
                end_calls=1,
                param_end_calls=param_end_calls,
                buffer_end_calls=buffer_end_calls,
                end_wall_time_sec=time.perf_counter() - t0,
            )

            if param_not_offload:
                if isinstance(output, torch.Tensor):
                    return output.to(torch.device(self.device))

                return copy_args_to_device(torch.device(self.device), output)

        # Pre forward hook
        self.forward_hooks.append(
            module.register_forward_pre_hook(
                _pre_forward_module_hook, with_kwargs=True)
        )

        # Post forward hook
        self.forward_hooks.append(
            module.register_forward_hook(_post_forward_module_hook)
        )

    # clean runtime hooks
    def clean_up(self):
        self.__exit__(None, None, None)
