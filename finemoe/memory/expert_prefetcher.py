# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

import torch
from transformers import PretrainedConfig
from finemoe.utils import parse_moe_param


class ExpertPrefetcher(object):
    cache_file_rd = None

    def __init__(self, config: PretrainedConfig, device):
        # print(config)
        self.num_layers, self.num_experts, self.num_encoder_layers, self.embed_dim, self.top_k = parse_moe_param(
            config)

        self.archer_engine = None
        self.expert_tensor_map = None
        self.resident_expert_ids = set()
        # [L, E] -> tensor_id (int64), -1 if missing
        self._tensor_id_grid = None
        # used only if you pass GPU tensors in
        self.device = torch.device(device)

    def set_archer_engine(self, archer_engine):
        if archer_engine is None:
            raise ValueError(
                "archer_engine must not be None. Call set_archer_engine(...) with a valid engine.")
        global _expert_prefetcher
        _expert_prefetcher = archer_engine
        self.archer_engine = archer_engine

    def set_expert_tensor_map(self, expert_tensor_map: dict):
        self.expert_tensor_map = expert_tensor_map
        self._build_tensor_id_grid()

    def set_resident_experts(self, expert_ids):
        self.resident_expert_ids = {
            (int(layer_id), int(expert_id)) for layer_id, expert_id in expert_ids
        }

    def _build_tensor_id_grid(self):
        if self.expert_tensor_map is None:
            raise RuntimeError(
                "expert_tensor_map is not set; set ExpertPrefetcher.expert_tensor_map or call set_expert_tensor_map(...).")
        grid = torch.full((self.num_layers, self.num_experts), -
                          1, dtype=torch.long, device="cpu")
        if self.expert_tensor_map is None:
            raise RuntimeError(
                "expert_tensor_map is not set on ExpertPrefetcher.")
        for (layer_id, expert_id), tid in self.expert_tensor_map.items():
            if 0 <= layer_id < self.num_layers and 0 <= expert_id < self.num_experts:
                grid[layer_id, expert_id] = int(tid)
        self._tensor_id_grid = grid

    @torch.inference_mode()
    def prefetch_experts(self, prefetch_priority_map, expert_prob_map):
        if self.archer_engine is None:
            raise RuntimeError(
                "ExpertPrefetcher.archer_engine is None. Call set_archer_engine(...) before prefetch_experts().")

        if not isinstance(prefetch_priority_map, torch.Tensor):
            prefetch_priority_map = torch.as_tensor(prefetch_priority_map)
        if not isinstance(expert_prob_map, torch.Tensor):
            expert_prob_map = torch.as_tensor(expert_prob_map)

        pp = prefetch_priority_map.detach().to(
            "cpu", dtype=torch.float32, non_blocking=False)
        ep = expert_prob_map.detach().to(
            "cpu", dtype=torch.float32, non_blocking=False)

        if self.resident_expert_ids:
            for layer_id, expert_id in self.resident_expert_ids:
                if 0 <= layer_id < pp.shape[0] and 0 <= expert_id < pp.shape[1]:
                    pp[layer_id, expert_id] = 0.0

        if self._tensor_id_grid is None:
            if getattr(self, "expert_tensor_map", None) is None:
                raise RuntimeError(
                    "expert_tensor_map not set; cannot map (layer, expert) to tensor ids.")
            self._build_tensor_id_grid()

        mask = pp > 0
        if not mask.any():
            return

        rows, cols = mask.nonzero(as_tuple=True)
        priors = pp[rows, cols]
        probs = ep[rows, cols]
        tids = self._tensor_id_grid[rows, cols]

        valid = tids >= 0
        if not valid.any():
            return

        tids = tids[valid]
        priors = priors[valid]
        probs = probs[valid]

        order = torch.argsort(priors, descending=True)
        tids = tids[order].tolist()
        probs = probs[order].tolist()

        self.archer_engine.replace_cache_candidates(tids)
        for tid, p in zip(tids, probs):
            gpu_id = self.archer_engine.get_node_default_device([tid])
            self.archer_engine.enqueue_prefetch(tid, gpu_id, float(p))

    @torch.inference_mode()
    def batch_prefetch_next_layer(self, current_layer, batch_expert_probs):
        """Batch-aware prefetch: one call for the entire batch.

        Instead of per-sequence embed/traj matching (heavy CPU overhead),
        this uses the batch-aggregated routing probabilities to prefetch
        experts for the next layer in a single pass.

        batch-aware 预取：一次调用覆盖整个 batch。
        用 batch 聚合的路由概率为下一层预取 expert，
        替代逐序列的 embed/traj 匹配（CPU 开销大）。

        Args:
            current_layer: index of the layer that just finished routing
            batch_expert_probs: [num_experts] tensor, max probability
                                across all tokens in the batch for each expert
        """
        if self.archer_engine is None:
            return
        if self._tensor_id_grid is None:
            return

        next_layer = current_layer + 1
        if next_layer >= self.num_layers:
            return

        probs = batch_expert_probs.detach().to("cpu", dtype=torch.float32)

        # Zero out resident experts (already on GPU)
        # 常驻 expert 已在 GPU 上，跳过
        for layer_id, expert_id in self.resident_expert_ids:
            if layer_id == next_layer and 0 <= expert_id < probs.shape[0]:
                probs[expert_id] = 0.0

        # Select experts above threshold (top_k probability > 0)
        # 选择概率大于 0 的 expert
        mask = probs > 0
        if not mask.any():
            return

        expert_ids = mask.nonzero(as_tuple=True)[0]
        expert_probs = probs[expert_ids]
        tids = self._tensor_id_grid[next_layer, expert_ids]

        valid = tids >= 0
        if not valid.any():
            return

        tids = tids[valid]
        expert_probs = expert_probs[valid]

        # Sort by probability descending, enqueue
        # 按概率降序排列，入队
        order = torch.argsort(expert_probs, descending=True)
        tids = tids[order].tolist()
        expert_probs = expert_probs[order].tolist()

        self.archer_engine.replace_cache_candidates(tids)
        for tid, p in zip(tids, expert_probs):
            gpu_id = self.archer_engine.get_node_default_device([tid])
            self.archer_engine.enqueue_prefetch(tid, gpu_id, float(p))
