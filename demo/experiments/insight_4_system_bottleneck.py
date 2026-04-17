"""
Insight 4: 系统瓶颈会在带宽受限区间发生相变
二维 sweep (device_memory_ratio × prefetch_window) 找到性能相变点

Metrics:
- p50/p95 latency (模拟)
- H2D bandwidth usage (模拟)
- GPU idle ratio (模拟)
- Cache hit rate
- Throughput (tokens/sec)
"""

import torch
import numpy as np
import pickle
import json
from pathlib import Path
from collections import defaultdict, OrderedDict


class SystemBottleneckAnalyzer:
    def __init__(
        self,
        state_file,
        mode='causal',
        predictor='history_freq',
        output_dir="./experiments/results",
        expert_size_mb=50.0,
        h2d_bandwidth_gbps=16.0,
        gpu_compute_time_ms=2.0,
        prefetch_admission='none',
        deadline_margin_ms=0.0,
        value_cost_scale=1.0,
        cache_layout='single',
        resident_ratio=0.5,
        resident_policy='none',
        resident_profile_ratio=0.1,
        resident_depth_power=1.0,
        output_tag="",
    ):
        self.state_file = state_file
        self.mode = mode  # 'causal' or 'oracle'
        self.predictor = 'oracle' if mode == 'oracle' else predictor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_tag = output_tag.strip()

        with open(state_file, 'rb') as f:
            self.trace_data = pickle.load(f)

        # 模拟参数
        self.expert_size_mb = float(expert_size_mb)
        self.h2d_bandwidth_gbps = float(h2d_bandwidth_gbps)  # PCIe 4.0 x16 = 16GB/s
        self.gpu_compute_time_ms = float(gpu_compute_time_ms)
        self.prefetch_admission = prefetch_admission
        self.deadline_margin_ms = float(deadline_margin_ms)
        self.value_cost_scale = float(value_cost_scale)
        self.cache_layout = cache_layout
        self.resident_ratio = float(resident_ratio)
        self.resident_policy = resident_policy
        self.resident_profile_ratio = float(resident_profile_ratio)
        self.resident_depth_power = float(resident_depth_power)

        # 提取访问序列
        self.access_sequence = self._extract_access_sequence()
        self.num_layers = self.trace_data[next(iter(self.trace_data))]['matrix'].shape[0]
        self.layer_compute_time_ms = self.gpu_compute_time_ms / max(1, self.num_layers)
        self.expert_access_count = self._count_expert_accesses()
        self._resident_selection_cache = {}

    def _transfer_time_ms(self):
        """单个 expert 的 H2D 传输时间，单位 ms。"""
        return (self.expert_size_mb / (self.h2d_bandwidth_gbps * 1024.0)) * 1000.0

    @staticmethod
    def _float_tag(value):
        return str(value).replace(".", "p")

    @staticmethod
    def _serialize_expert_pairs(expert_keys):
        return [
            {'layer': int(layer_idx), 'expert': int(expert_idx)}
            for layer_idx, expert_idx in sorted(expert_keys)
        ]

    @staticmethod
    def _serialize_ranked_expert_pairs(ranked_items):
        return [
            {'layer': int(layer_idx), 'expert': int(expert_idx)}
            for (layer_idx, expert_idx), _ in ranked_items
        ]

    @staticmethod
    def _serialize_expert_metric(metric_dict):
        return [
            {'layer': int(layer_idx), 'expert': int(expert_idx), 'value': float(value)}
            for (layer_idx, expert_idx), value in sorted(metric_dict.items())
        ]

    def get_resident_set(self, device_memory_ratio, reset_mode='shared'):
        total_gpu_memory_mb = 80 * 1024
        available_memory_mb = total_gpu_memory_mb * device_memory_ratio
        cache_capacity = int(available_memory_mb / self.expert_size_mb)
        resident_capacity, speculative_capacity = self._pool_capacities(cache_capacity)
        ranked_resident = self._rank_resident_experts(
            resident_capacity, cache_capacity, reset_mode
        )
        resident_set = self._select_resident_experts(resident_capacity, cache_capacity, reset_mode)
        return {
            'device_memory_ratio': float(device_memory_ratio),
            'reset_mode': reset_mode,
            'cache_capacity': int(cache_capacity),
            'resident_capacity': int(resident_capacity),
            'speculative_capacity': int(speculative_capacity),
            'resident_policy': self.resident_policy,
            'resident_profile_ratio': self.resident_profile_ratio,
            'resident_depth_power': self.resident_depth_power,
            'resident_set': self._serialize_expert_pairs(resident_set),
            'resident_selection_order': self._serialize_ranked_expert_pairs(ranked_resident[:resident_capacity]),
            'resident_ordered': True,
        }

    def _result_file(self, reset_mode):
        parts = [f"insight_4_mode{self.mode}"]
        if self.mode != 'oracle':
            parts.append(f"predictor{self.predictor}")
        parts.append(f"reset{reset_mode}")
        if self.prefetch_admission != 'none':
            parts.append(f"adm{self.prefetch_admission}")
        if abs(self.deadline_margin_ms) > 1e-9:
            parts.append(f"margin{self._float_tag(self.deadline_margin_ms)}")
        if self.cache_layout != 'single':
            parts.append(f"layout{self.cache_layout}")
            parts.append(f"res{self._float_tag(self.resident_ratio)}")
            if self.resident_policy != 'none':
                parts.append(f"rpol{self.resident_policy}")
            if self.resident_policy in {'profile_freq', 'profile_depth_freq', 'profile_miss_stall'}:
                parts.append(f"rprof{self._float_tag(self.resident_profile_ratio)}")
            if self.resident_policy == 'profile_depth_freq' and abs(self.resident_depth_power - 1.0) > 1e-9:
                parts.append(f"rdpow{self._float_tag(self.resident_depth_power)}")
        if abs(self.expert_size_mb - 50.0) > 1e-9:
            parts.append(f"e{self._float_tag(self.expert_size_mb)}")
        if self.output_tag:
            parts.append(self.output_tag)
        return self.output_dir / ("_".join(parts) + ".json")

    def _pool_capacities(self, cache_capacity):
        if self.cache_layout != 'two_pool':
            return cache_capacity, 0
        if cache_capacity <= 0:
            return 0, 0
        if cache_capacity == 1:
            return 1, 0
        resident_capacity = int(round(cache_capacity * self.resident_ratio))
        resident_capacity = max(1, min(cache_capacity - 1, resident_capacity))
        speculative_capacity = max(0, cache_capacity - resident_capacity)
        return resident_capacity, speculative_capacity

    def _layers_until_due(self, token_idx, layer_idx, due_token_idx, due_layer_idx):
        """从当前 layer compute 开始，到目标请求开始前还剩多少个 layer-compute 时隙。"""
        if due_token_idx < token_idx:
            return 0
        if due_token_idx == token_idx:
            return max(0, due_layer_idx - layer_idx)
        return (
            (self.num_layers - layer_idx)
            + (due_token_idx - token_idx - 1) * self.num_layers
            + due_layer_idx
        )

    def _estimate_due_time(self, token_idx, layer_idx, due_token_idx, due_layer_idx, compute_start_time):
        layer_slots = self._layers_until_due(token_idx, layer_idx, due_token_idx, due_layer_idx)
        return compute_start_time + layer_slots * self.layer_compute_time_ms

    def _estimate_prefetch_timing(
        self,
        token_idx,
        layer_idx,
        request,
        compute_start_time,
        prefetch_start_time,
        h2d_state,
    ):
        transfer_time_ms = self._transfer_time_ms()
        due_time = self._estimate_due_time(
            token_idx,
            layer_idx,
            request['due_token_idx'],
            request['due_layer_idx'],
            compute_start_time,
        )
        busy_until = h2d_state['available_time']
        candidate_start = max(prefetch_start_time, busy_until)
        candidate_ready = candidate_start + transfer_time_ms
        demand_start = max(due_time, busy_until)
        demand_ready = demand_start + transfer_time_ms
        prefetch_stall = max(0.0, candidate_ready - due_time)
        demand_stall = demand_ready - due_time
        saved_stall = max(0.0, demand_stall - prefetch_stall)
        queue_wait = max(0.0, busy_until - prefetch_start_time)
        return {
            'transfer_time_ms': transfer_time_ms,
            'due_time': due_time,
            'candidate_start': candidate_start,
            'candidate_ready': candidate_ready,
            'demand_ready': demand_ready,
            'prefetch_stall': prefetch_stall,
            'demand_stall': demand_stall,
            'saved_stall': saved_stall,
            'queue_wait': queue_wait,
        }

    def _prefetch_value_score(
        self,
        token_idx,
        layer_idx,
        request,
        compute_start_time,
        prefetch_start_time,
        cache,
        inflight,
        h2d_state,
    ):
        """
        Value-aware admission score:
            expected_stall_saved - queue_cost

        All terms are estimated from the current simulator state, without looking ahead.
        """
        timing = self._estimate_prefetch_timing(
            token_idx,
            layer_idx,
            request,
            compute_start_time,
            prefetch_start_time,
            h2d_state,
        )
        confidence = float(request.get('normalized_score', request.get('score', 1.0)))
        depth_factor = (request['due_layer_idx'] + 1) / max(1, self.num_layers)
        cache_capacity = max(1, h2d_state.get('cache_capacity', 1))
        pressure = (len(cache) + len(inflight)) / float(cache_capacity)
        queue_cost = self.value_cost_scale * (
            timing['queue_wait'] + pressure * timing['transfer_time_ms']
        )
        benefit = confidence * depth_factor * timing['saved_stall']
        return {
            **timing,
            'confidence': confidence,
            'depth_factor': depth_factor,
            'pressure': pressure,
            'queue_cost': queue_cost,
            'benefit': benefit,
            'value_score': benefit - queue_cost,
        }

    def _prefetch_admission_check(
        self,
        token_idx,
        layer_idx,
        request,
        compute_start_time,
        prefetch_start_time,
        cache,
        inflight,
        h2d_state,
    ):
        """
        决定某个预测请求是否值得进入 H2D 队列。

        `deadline` admission 直接拒绝那些按当前队列状态看必然晚到的预取。
        """
        timing = self._estimate_prefetch_timing(
            token_idx,
            layer_idx,
            request,
            compute_start_time,
            prefetch_start_time,
            h2d_state,
        )
        candidate_ready = timing['candidate_ready']
        due_time = timing['due_time']
        if self.prefetch_admission == 'deadline':
            latest_allowed_ready = due_time - self.deadline_margin_ms
            return candidate_ready <= latest_allowed_ready, candidate_ready, due_time, timing
        if self.prefetch_admission == 'value':
            value_info = self._prefetch_value_score(
                token_idx,
                layer_idx,
                request,
                compute_start_time,
                prefetch_start_time,
                cache,
                inflight,
                h2d_state,
            )
            return value_info['value_score'] > 0.0, candidate_ready, due_time, value_info
        return True, candidate_ready, due_time, timing

    def _resident_occupancy(self, cache, speculative_entries):
        return len(cache) - len(speculative_entries)

    def _speculative_occupancy(self, speculative_entries):
        return len(speculative_entries)

    def _inflight_occupancy(self, inflight, transfer_kind):
        return sum(1 for state in inflight.values() if state['kind'] == transfer_kind)

    def _promote_to_resident(
        self,
        expert_key,
        cache,
        inflight,
        speculative_entries,
        resident_capacity,
        protected_keys,
    ):
        if expert_key not in speculative_entries:
            return True
        speculative_entries.discard(expert_key)
        cache.move_to_end(expert_key)
        return True

    def _materialize_ready_experts(self, current_time, cache, inflight, speculative_entries):
        """把已经传输完成的 expert 从 inflight 转入 cache。"""
        ready_keys = [key for key, state in inflight.items() if state['ready_time'] <= current_time]
        for key in ready_keys:
            cache[key] = True
            cache.move_to_end(key)
            if inflight[key]['kind'] == 'prefetch':
                speculative_entries.add(key)
            else:
                speculative_entries.discard(key)
            del inflight[key]

    def _evict_one(self, cache, inflight, speculative_entries, protected_keys, pool_filter='any'):
        """
        为新 expert 腾出一个 slot。

        优先驱逐 resident LRU；如果 resident 都受保护，再取消一个最晚完成的 inflight 预取。
        """
        for key in list(cache.keys()):
            if pool_filter == 'resident' and key in speculative_entries:
                continue
            if pool_filter == 'speculative' and key not in speculative_entries:
                continue
            if key not in protected_keys:
                del cache[key]
                speculative_entries.discard(key)
                return True

        cancel_candidates = [
            (state['ready_time'], key)
            for key, state in inflight.items()
            if key not in protected_keys
            and state['kind'] == 'prefetch'
            and (pool_filter != 'resident')
        ]
        if cancel_candidates:
            _, victim = max(cancel_candidates)
            del inflight[victim]
            return True

        return False

    def _ensure_capacity(self, h2d_state, cache, inflight, speculative_entries, protected_keys, transfer_kind):
        """确保相应 pool 的占用不超过容量。"""
        cache_capacity = h2d_state.get("cache_capacity", 0)
        if cache_capacity <= 0:
            return False

        if self.cache_layout != 'two_pool':
            while len(cache) + len(inflight) >= cache_capacity:
                if not self._evict_one(cache, inflight, speculative_entries, protected_keys, pool_filter='any'):
                    return False
            return True

        speculative_capacity = h2d_state.get("speculative_capacity", 0)
        if transfer_kind == 'prefetch':
            if speculative_capacity <= 0:
                return False
            while (
                self._speculative_occupancy(speculative_entries)
                + self._inflight_occupancy(inflight, 'prefetch')
                >= speculative_capacity
            ):
                if not self._evict_one(
                    cache,
                    inflight,
                    speculative_entries,
                    protected_keys,
                    pool_filter='speculative',
                ):
                    return False
        else:
            while len(cache) + len(inflight) >= cache_capacity:
                if self._evict_one(
                    cache,
                    inflight,
                    speculative_entries,
                    protected_keys,
                    pool_filter='speculative',
                ):
                    continue
                if not self._evict_one(
                    cache,
                    inflight,
                    speculative_entries,
                    protected_keys,
                    pool_filter='resident',
                ):
                    return False
        return True

    def _schedule_transfer(
        self,
        expert_key,
        earliest_start_time,
        transfer_kind,
        cache,
        inflight,
        speculative_entries,
        h2d_state,
        h2d_bytes,
        protected_keys,
    ):
        """
        在共享 H2D 队列上排队一个 expert 传输。

        返回: (scheduled: bool, ready_time: float, new_h2d_bytes: int)
        """
        if expert_key in cache:
            return True, earliest_start_time, h2d_bytes
        if expert_key in inflight:
            return True, inflight[expert_key]['ready_time'], h2d_bytes

        if not self._ensure_capacity(
            h2d_state,
            cache,
            inflight,
            speculative_entries,
            protected_keys,
            transfer_kind,
        ):
            return False, earliest_start_time, h2d_bytes

        transfer_time_ms = self._transfer_time_ms()
        start_time = max(earliest_start_time, h2d_state["available_time"])
        ready_time = start_time + transfer_time_ms
        inflight[expert_key] = {
            'ready_time': ready_time,
            'kind': transfer_kind,
        }
        h2d_state["available_time"] = ready_time
        h2d_bytes += self.expert_size_mb * 1024 * 1024
        return True, ready_time, h2d_bytes

    def _count_expert_accesses(self, profile_ratio=1.0, score_mode='freq'):
        expert_freq = defaultdict(float)
        num_tokens = len(self.access_sequence)
        limit = num_tokens
        if profile_ratio < 1.0:
            limit = max(1, min(num_tokens, int(round(num_tokens * profile_ratio))))
        for token_data in self.access_sequence[:limit]:
            for expert_key in token_data['experts']:
                if score_mode == 'freq':
                    weight = 1.0
                elif score_mode == 'depth_freq':
                    depth = (expert_key[0] + 1) / max(1, self.num_layers)
                    weight = depth ** self.resident_depth_power
                else:
                    raise ValueError(f"Unsupported score_mode: {score_mode}")
                expert_freq[expert_key] += weight
        return dict(expert_freq)

    def _profile_miss_stall_scores(self, profile_ratio, cache_capacity, resident_capacity, reset_mode):
        """
        用 prefix trace 在 demand-only 条件下直接估计每个 expert 的 stall contribution。

        这里的 dynamic cache 大小取 `cache_capacity - resident_capacity`，
        因为 resident pool 占掉的容量本质上是在牺牲动态缓存空间来换确定性。
        """
        dynamic_cache_capacity = max(1, cache_capacity - resident_capacity)
        num_tokens = len(self.access_sequence)
        limit = max(1, min(num_tokens, int(round(num_tokens * profile_ratio))))
        sequence = self.access_sequence[:limit]

        cache = OrderedDict()
        inflight = {}
        current_time = 0.0
        h2d_available_time = 0.0
        current_seq_id = None
        scores = defaultdict(float)

        transfer_time_ms = self._transfer_time_ms()

        for token_data in sequence:
            if reset_mode == 'per_sequence' and token_data['seq_id'] != current_seq_id:
                cache.clear()
                inflight.clear()
                current_seq_id = token_data['seq_id']
                h2d_available_time = current_time

            for layer_idx, layer_experts in enumerate(token_data['layer_experts']):
                ready_keys = [key for key, state in inflight.items() if state['ready_time'] <= current_time]
                for key in ready_keys:
                    cache[key] = True
                    cache.move_to_end(key)
                    del inflight[key]

                layer_ready_time = current_time
                protected_keys = set(layer_experts)

                for expert_key in layer_experts:
                    if expert_key in cache:
                        cache.move_to_end(expert_key)
                        continue

                    if expert_key in inflight:
                        ready_time = inflight[expert_key]['ready_time']
                        if ready_time <= current_time:
                            del inflight[expert_key]
                            cache[expert_key] = True
                            cache.move_to_end(expert_key)
                        else:
                            stall = ready_time - current_time
                            scores[expert_key] += stall
                            layer_ready_time = max(layer_ready_time, ready_time)
                        continue

                    while len(cache) + len(inflight) >= dynamic_cache_capacity:
                        victim = None
                        for key in cache.keys():
                            if key not in protected_keys:
                                victim = key
                                break
                        if victim is None:
                            break
                        del cache[victim]

                    start_time = max(current_time, h2d_available_time)
                    ready_time = start_time + transfer_time_ms
                    inflight[expert_key] = {'ready_time': ready_time}
                    h2d_available_time = ready_time
                    scores[expert_key] += ready_time - current_time
                    layer_ready_time = max(layer_ready_time, ready_time)

                ready_keys = [key for key, state in inflight.items() if state['ready_time'] <= layer_ready_time]
                for key in ready_keys:
                    cache[key] = True
                    cache.move_to_end(key)
                    del inflight[key]

                current_time = layer_ready_time + self.layer_compute_time_ms

        return dict(scores)

    def _rank_resident_experts(self, resident_capacity, cache_capacity, reset_mode):
        """
        返回 resident 候选的有序排名，供 resident pool 选择和 runtime 裁剪复用。

        `oracle_freq` 使用全 trace 访问频率做一个 oracle-labeled upper-bound，
        用来验证 two-pool 结构本身是否值得继续做。
        """
        if self.cache_layout != 'two_pool' or resident_capacity <= 0:
            return []
        if self.resident_policy == 'none':
            return []
        cache_key = (
            "ranked",
            self.resident_policy,
            resident_capacity,
            cache_capacity,
            reset_mode,
            self.resident_profile_ratio,
            self.resident_depth_power,
        )
        if cache_key in self._resident_selection_cache:
            return self._resident_selection_cache[cache_key]
        if self.resident_policy == 'oracle_freq':
            ranked = sorted(
                self.expert_access_count.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
            self._resident_selection_cache[cache_key] = ranked
            return ranked
        if self.resident_policy == 'profile_freq':
            profiled_access_count = self._count_expert_accesses(self.resident_profile_ratio)
            ranked = sorted(
                profiled_access_count.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
            self._resident_selection_cache[cache_key] = ranked
            return ranked
        if self.resident_policy == 'profile_depth_freq':
            profiled_access_count = self._count_expert_accesses(
                self.resident_profile_ratio,
                score_mode='depth_freq',
            )
            ranked = sorted(
                profiled_access_count.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
            self._resident_selection_cache[cache_key] = ranked
            return ranked
        if self.resident_policy == 'profile_miss_stall':
            profiled_scores = self._profile_miss_stall_scores(
                self.resident_profile_ratio,
                cache_capacity,
                resident_capacity,
                reset_mode,
            )
            ranked = sorted(
                profiled_scores.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
            self._resident_selection_cache[cache_key] = ranked
            return ranked
        raise ValueError(f"Unsupported resident_policy: {self.resident_policy}")

    def _select_resident_experts(self, resident_capacity, cache_capacity, reset_mode):
        ranked = self._rank_resident_experts(
            resident_capacity,
            cache_capacity,
            reset_mode,
        )
        return {expert_key for expert_key, _ in ranked[:resident_capacity]}

    @staticmethod
    def _seed_resident_cache(cache, speculative_entries, pinned_resident_keys):
        cache.clear()
        speculative_entries.clear()
        for expert_key in sorted(pinned_resident_keys):
            cache[expert_key] = True

    def _extract_access_sequence(self):
        """提取按 token/layer 组织的 expert 访问序列。"""
        sequence = []
        for seq_id, trace_entry in self.trace_data.items():
            iters = trace_entry['iters']
            num_layers = trace_entry['matrix'].shape[0]

            for iter_idx, it in enumerate(iters):
                layer_experts = []
                token_experts = []
                nodes = it['nodes']
                for layer_idx in range(num_layers):
                    experts = [(layer_idx, e) for e in torch.nonzero(nodes[layer_idx] > 0).squeeze(-1).tolist()]
                    layer_experts.append(experts)
                    token_experts.extend(experts)
                sequence.append({
                    'seq_id': seq_id,
                    'iter_idx': iter_idx,
                    'layer_experts': layer_experts,
                    'experts': token_experts,
                })
        return sequence

    def _iter_future_requests(self, token_idx, layer_idx, window):
        """枚举从当前位置之后、窗口内的未来 layer 请求。"""
        if window <= 0:
            return []

        requests = []
        max_token_idx = min(len(self.access_sequence) - 1, token_idx + window)
        for future_token_idx in range(token_idx, max_token_idx + 1):
            start_layer = layer_idx + 1 if future_token_idx == token_idx else 0
            for future_layer_idx in range(start_layer, self.num_layers):
                for expert_key in self.access_sequence[future_token_idx]['layer_experts'][future_layer_idx]:
                    requests.append({
                        'expert_key': expert_key,
                        'due_token_idx': future_token_idx,
                        'due_layer_idx': future_layer_idx,
                        'score': 1.0,
                    })
        return requests

    def _iter_past_requests(self, token_idx, layer_idx, window):
        """枚举当前位置之前因果可见的历史 layer 请求。"""
        if token_idx == 0 and layer_idx == 0:
            return []

        start_token_idx = max(0, token_idx - max(1, window))
        requests = []
        for past_token_idx in range(start_token_idx, token_idx + 1):
            end_layer = layer_idx if past_token_idx == token_idx else self.num_layers
            for past_layer_idx in range(end_layer):
                for expert_key in self.access_sequence[past_token_idx]['layer_experts'][past_layer_idx]:
                    requests.append({
                        'expert_key': expert_key,
                        'token_idx': past_token_idx,
                        'layer_idx': past_layer_idx,
                    })
        return requests

    def _predict_future_requests_history_freq(self, token_idx, layer_idx, window):
        """
        因果预测：只使用当前位置之前的信息预测未来 layer 请求。

        当前 baseline 仍然是轻量频率预测，但排序和 deadline 都在 layer-aware 坐标系里。
        """
        if window <= 0:
            return []

        expert_freq = defaultdict(int)
        for request in self._iter_past_requests(token_idx, layer_idx, window):
            expert_freq[request['expert_key']] += 1

        if not expert_freq:
            return []

        predicted = []
        top_k = 4 * window
        ranked = sorted(expert_freq.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))[:top_k]
        for expert_key, score in ranked:
            predicted.append({
                'expert_key': expert_key,
                'due_token_idx': token_idx if expert_key[0] > layer_idx else token_idx + 1,
                'due_layer_idx': expert_key[0],
                'score': float(score),
            })

        predicted.sort(key=lambda item: (item['due_token_idx'], item['due_layer_idx'], -item['score']))
        return predicted

    def _predict_future_requests_pl_ctr(self, token_idx, layer_idx, window):
        """
        PL-CTR: 用当前 token 在同一层的 experts 预测下一 token 在该层的 experts。

        这是一个严格 layer-local 的 consecutive-token repeat baseline，不做跨层复用。
        当跨越 sequence 边界或当前层无 expert 时，退回 history-frequency baseline。
        """
        if window <= 0:
            return []

        next_token_idx = token_idx + 1
        if next_token_idx >= len(self.access_sequence):
            return self._predict_future_requests_history_freq(token_idx, layer_idx, window)

        current_token = self.access_sequence[token_idx]
        next_token = self.access_sequence[next_token_idx]
        if current_token['seq_id'] != next_token['seq_id']:
            return self._predict_future_requests_history_freq(token_idx, layer_idx, window)

        experts = current_token['layer_experts'][layer_idx]
        if not experts:
            return self._predict_future_requests_history_freq(token_idx, layer_idx, window)

        return [
            {
                'expert_key': expert_key,
                'due_token_idx': next_token_idx,
                'due_layer_idx': layer_idx,
                'score': 1.0,
            }
            for expert_key in experts
        ]

    def _predict_future_requests_utility_freq(self, token_idx, layer_idx, window):
        """
        Utility-aware prefetch: directly optimize U(e) = P(not resident) * avoidable_stall.

        U(e) = staleness(e) * frequency(e) * depth_factor(layer_of(e))

        Unlike history_freq (optimizes P(access)) or PL-CTR (optimizes P(access|same layer)),
        this targets experts that are likely to MISS — the sweet spot for prefetch.

        The key insight: hot experts (history_freq target) are already in cache (P(not resident) ≈ 0).
        Cold experts (never accessed) have P(access) ≈ 0. The sweet spot is
        COLD-BUT-FREQUENT: recently accessed but not in current working set.

        staleness: how long since last access (normalized by lookback window)
        frequency: how often accessed in lookback window
        depth_factor: deeper layers have more stall impact
        """
        if window <= 0:
            return []

        lookback = max(1, window)
        start_token_idx = max(0, token_idx - lookback)

        # Collect frequency and last-seen from history
        expert_freq = defaultdict(int)
        expert_last_seen = {}
        for past_token_idx in range(start_token_idx, token_idx + 1):
            past_end_layer = layer_idx if past_token_idx == token_idx else self.num_layers
            for past_layer_idx in range(past_end_layer):
                for expert_key in self.access_sequence[past_token_idx]['layer_experts'][past_layer_idx]:
                    expert_freq[expert_key] += 1
                    expert_last_seen[expert_key] = past_token_idx

        if not expert_freq:
            return []

        max_freq = max(expert_freq.values())

        # Score each candidate: U(e) = staleness * freq * depth_factor
        # staleness = (token_idx - last_seen) / lookback  → higher = more likely evicted
        # freq_normalized = freq / max_freq               → higher = more likely accessed again
        # depth_factor = (layer / num_layers)             → higher = more stall impact
        candidates = []
        for expert_key, freq in expert_freq.items():
            staleness = (token_idx - expert_last_seen[expert_key]) / float(lookback)
            freq_norm = freq / float(max_freq)
            depth_factor = expert_key[0] / max(1, self.num_layers - 1)
            utility_score = staleness * freq_norm * depth_factor
            candidates.append((expert_key, utility_score))

        # Sort by utility descending; within same score, prefer deeper layers
        candidates.sort(key=lambda item: (-item[1], -item[0][0], item[0][1]))

        # Top-K: scale with window size, cap at 4*window
        top_k = min(4 * window, len(candidates))
        predicted = []
        for expert_key, score in candidates[:top_k]:
            due_token = token_idx if expert_key[0] > layer_idx else token_idx + 1
            due_layer = expert_key[0]
            predicted.append({
                'expert_key': expert_key,
                'due_token_idx': due_token,
                'due_layer_idx': due_layer,
                'score': float(score),
            })

        predicted.sort(key=lambda item: (item['due_token_idx'], item['due_layer_idx'], -item['score']))
        return predicted

    def _predict_future_requests_oracle(self, token_idx, layer_idx, window):
        """
        Oracle 预测：直接查看未来 token/layer 的真实请求，作为上界对比。
        """
        first_occurrence = {}
        for request in self._iter_future_requests(token_idx, layer_idx, window):
            first_occurrence.setdefault(request['expert_key'], request)
        return sorted(first_occurrence.values(), key=lambda item: (item['due_token_idx'], item['due_layer_idx']))

    def _predict_future_requests(self, token_idx, layer_idx, window):
        """根据 mode 选择预测方法"""
        if self.mode == 'oracle':
            return self._predict_future_requests_oracle(token_idx, layer_idx, window)
        if self.predictor == 'pl_ctr':
            return self._predict_future_requests_pl_ctr(token_idx, layer_idx, window)
        if self.predictor == 'utility_freq':
            return self._predict_future_requests_utility_freq(token_idx, layer_idx, window)
        return self._predict_future_requests_history_freq(token_idx, layer_idx, window)

    def simulate_with_config(self, device_memory_ratio, prefetch_window, reset_mode='shared'):
        """
        Layer-aware 事件驱动模拟：按 token x layer 推进时间线。

        device_memory_ratio: GPU 显存使用比例 (0.1-1.0)
        prefetch_window: prefetch 窗口大小 (0-10)
        reset_mode: 'per_sequence' (每序列重置) 或 'shared' (跨序列共享)
        """
        # 计算 cache capacity (基于 device_memory_ratio)
        # 假设 A800 80GB，每个 expert 50MB
        total_gpu_memory_mb = 80 * 1024
        available_memory_mb = total_gpu_memory_mb * device_memory_ratio
        cache_capacity = int(available_memory_mb / self.expert_size_mb)
        resident_capacity, speculative_capacity = self._pool_capacities(cache_capacity)
        pinned_resident_keys = self._select_resident_experts(resident_capacity, cache_capacity, reset_mode)

        # 模拟 cache (LRU)
        cache = OrderedDict()
        speculative_entries = set()
        self._seed_resident_cache(cache, speculative_entries, pinned_resident_keys)

        # Expert 状态跟踪
        inflight = {}  # expert_key -> {'ready_time': float, 'kind': str}

        # 统计指标
        latencies = []
        h2d_bytes = 0
        gpu_idle_time = 0
        total_compute_time = 0
        hits = 0
        misses = 0
        prefetch_hits = 0
        late_prefetches = 0
        total_residual_stall_ms = 0.0

        per_layer_hits = [0 for _ in range(self.num_layers)]
        per_layer_misses = [0 for _ in range(self.num_layers)]
        per_layer_prefetch_hits = [0 for _ in range(self.num_layers)]
        per_layer_late_prefetches = [0 for _ in range(self.num_layers)]
        per_layer_residual_stall_ms = [0.0 for _ in range(self.num_layers)]
        per_expert_accesses = defaultdict(int)
        per_expert_critical_stall_ms = defaultdict(float)
        per_expert_demand_misses = defaultdict(int)
        per_expert_late_prefetches = defaultdict(int)

        # Funnel 跟踪: predicted -> novel -> scheduled -> timely/useful
        funnel_predicted = 0
        funnel_novel = 0
        funnel_scheduled = 0
        funnel_admission_dropped = 0
        funnel_deadline_rejected = 0
        funnel_value_rejected = 0
        funnel_timely = 0
        funnel_late = 0
        funnel_useful = 0
        funnel_useful_stall_saved_ms = 0.0
        tail_funnel_predicted = 0
        tail_funnel_novel = 0
        tail_funnel_scheduled = 0
        tail_funnel_timely = 0
        tail_funnel_late = 0
        tail_funnel_useful = 0
        tail_funnel_useful_stall_saved_ms = 0.0
        # prefetch_log[expert_key] = {'predicted_at_token': t, 'predicted_at_layer': l,
        #                              'was_novel_at_predict': bool, 'ready_time': rt}
        prefetch_log = {}

        # 事件时间
        current_time = 0.0
        h2d_state = {
            'available_time': 0.0,
            'cache_capacity': cache_capacity,
            'resident_capacity': resident_capacity,
            'speculative_capacity': speculative_capacity,
        }

        # 按序列分组（用于 reset_mode）
        current_seq_id = None

        for token_idx, token_data in enumerate(self.access_sequence):
            # Reset cache per sequence if needed
            if reset_mode == 'per_sequence' and token_data['seq_id'] != current_seq_id:
                self._seed_resident_cache(cache, speculative_entries, pinned_resident_keys)
                inflight.clear()
                prefetch_log.clear()
                current_seq_id = token_data['seq_id']
                h2d_state['available_time'] = current_time

            token_start_time = current_time
            for layer_idx, layer_experts in enumerate(token_data['layer_experts']):
                self._materialize_ready_experts(current_time, cache, inflight, speculative_entries)

                layer_ready_time = current_time
                protected_keys = set(layer_experts) | pinned_resident_keys
                late_prefetch_keys = set()

                for expert_key in layer_experts:
                    per_expert_accesses[expert_key] += 1

                    if expert_key in cache:
                        cache.move_to_end(expert_key)
                        if expert_key in speculative_entries:
                            prefetch_hits += 1
                            per_layer_prefetch_hits[layer_idx] += 1
                            plog = prefetch_log.pop(expert_key, None)
                            if plog is not None and plog['was_novel_at_predict']:
                                saved_stall = float(plog.get('saved_stall', self._transfer_time_ms()))
                                funnel_timely += 1
                                funnel_useful += 1
                                funnel_useful_stall_saved_ms += saved_stall
                                if plog.get('is_tail', False):
                                    tail_funnel_timely += 1
                                    tail_funnel_useful += 1
                                    tail_funnel_useful_stall_saved_ms += saved_stall
                            self._promote_to_resident(
                                expert_key,
                                cache,
                                inflight,
                                speculative_entries,
                                resident_capacity,
                                protected_keys,
                            )
                        else:
                            hits += 1
                            per_layer_hits[layer_idx] += 1
                        continue

                    if expert_key in inflight:
                        ready_time = inflight[expert_key]['ready_time']
                        plog = prefetch_log.get(expert_key)
                        if ready_time <= current_time:
                            if inflight[expert_key]['kind'] == 'prefetch':
                                prefetch_hits += 1
                                per_layer_prefetch_hits[layer_idx] += 1
                                if plog is not None and plog['was_novel_at_predict']:
                                    saved_stall = float(plog.get('saved_stall', self._transfer_time_ms()))
                                    funnel_timely += 1
                                    funnel_useful += 1
                                    funnel_useful_stall_saved_ms += saved_stall
                                    if plog.get('is_tail', False):
                                        tail_funnel_timely += 1
                                        tail_funnel_useful += 1
                                        tail_funnel_useful_stall_saved_ms += saved_stall
                                prefetch_log.pop(expert_key, None)
                                del inflight[expert_key]
                                cache[expert_key] = True
                                cache.move_to_end(expert_key)
                                self._promote_to_resident(
                                    expert_key,
                                    cache,
                                    inflight,
                                    speculative_entries,
                                    resident_capacity,
                                    protected_keys,
                                )
                            else:
                                hits += 1
                                per_layer_hits[layer_idx] += 1
                                del inflight[expert_key]
                                cache[expert_key] = True
                                cache.move_to_end(expert_key)
                                speculative_entries.discard(expert_key)
                        else:
                            late_prefetches += 1
                            per_layer_late_prefetches[layer_idx] += 1
                            per_expert_late_prefetches[expert_key] += 1
                            late_prefetch_keys.add(expert_key)
                            previous_ready_time = layer_ready_time
                            layer_ready_time = max(layer_ready_time, ready_time)
                            per_expert_critical_stall_ms[expert_key] += max(0.0, layer_ready_time - previous_ready_time)
                            if plog is not None and plog['was_novel_at_predict']:
                                funnel_late += 1
                                if plog.get('is_tail', False):
                                    tail_funnel_late += 1
                            prefetch_log.pop(expert_key, None)
                        continue

                    misses += 1
                    per_layer_misses[layer_idx] += 1
                    per_expert_demand_misses[expert_key] += 1
                    prefetch_log.pop(expert_key, None)
                    scheduled, ready_time, h2d_bytes = self._schedule_transfer(
                        expert_key,
                        current_time,
                        'demand',
                        cache,
                        inflight,
                        speculative_entries,
                        h2d_state,
                        h2d_bytes,
                        protected_keys,
                    )
                    if not scheduled:
                        raise RuntimeError(
                            f"Cannot reserve capacity for mandatory expert {expert_key} "
                            f"with cache_capacity={cache_capacity}"
                        )
                    previous_ready_time = layer_ready_time
                    layer_ready_time = max(layer_ready_time, ready_time)
                    per_expert_critical_stall_ms[expert_key] += max(0.0, layer_ready_time - previous_ready_time)

                self._materialize_ready_experts(layer_ready_time, cache, inflight, speculative_entries)
                for expert_key in late_prefetch_keys:
                    self._promote_to_resident(
                        expert_key,
                        cache,
                        inflight,
                        speculative_entries,
                        resident_capacity,
                        protected_keys,
                    )

                layer_stall = max(0.0, layer_ready_time - current_time)
                total_residual_stall_ms += layer_stall
                per_layer_residual_stall_ms[layer_idx] += layer_stall
                if layer_stall > 0:
                    gpu_idle_time += layer_stall

                compute_start_time = layer_ready_time
                total_compute_time += self.layer_compute_time_ms

                if prefetch_window > 0:
                    predicted_requests = self._predict_future_requests(token_idx, layer_idx, prefetch_window)
                    prefetch_start_time = max(compute_start_time, h2d_state['available_time'])
                    max_score = max((float(request.get('score', 1.0)) for request in predicted_requests), default=1.0)
                    score_denom = max(1e-9, max_score)
                    predict_cache_keys = set(cache.keys())
                    predict_inflight_keys = set(inflight.keys())
                    deduped_requests = []
                    seen_keys = set()
                    for request in predicted_requests:
                        expert_key = request['expert_key']
                        if expert_key in seen_keys:
                            continue
                        seen_keys.add(expert_key)
                        request_copy = dict(request)
                        request_copy['normalized_score'] = float(request.get('score', 1.0)) / score_denom
                        request_copy['is_tail'] = expert_key not in pinned_resident_keys
                        request_copy['was_novel_at_predict'] = (
                            expert_key not in predict_cache_keys and expert_key not in predict_inflight_keys
                        )
                        deduped_requests.append(request_copy)
                        funnel_predicted += 1
                        if request_copy['is_tail']:
                            tail_funnel_predicted += 1
                        if request_copy['was_novel_at_predict']:
                            funnel_novel += 1
                            if request_copy['is_tail']:
                                tail_funnel_novel += 1

                    novel_requests = [request for request in deduped_requests if request['was_novel_at_predict']]

                    if self.prefetch_admission == 'value':
                        pending_requests = list(novel_requests)
                        while pending_requests:
                            scored = []
                            for request in pending_requests:
                                expert_key = request['expert_key']
                                if expert_key in cache or expert_key in inflight:
                                    continue
                                value_info = self._prefetch_value_score(
                                    token_idx,
                                    layer_idx,
                                    request,
                                    compute_start_time,
                                    prefetch_start_time,
                                    cache,
                                    inflight,
                                    h2d_state,
                                )
                                scored.append((value_info['value_score'], request, value_info))

                            if not scored:
                                break

                            best_score, best_request, best_value_info = max(
                                scored,
                                key=lambda item: (
                                    item[0],
                                    -item[1]['due_token_idx'],
                                    -item[1]['due_layer_idx'],
                                    item[1].get('normalized_score', 0.0),
                                ),
                            )

                            pending_requests = [
                                request for request in pending_requests
                                if request['expert_key'] != best_request['expert_key']
                            ]

                            if best_score <= 0.0:
                                funnel_value_rejected += 1 + len(pending_requests)
                                break

                            expert_key = best_request['expert_key']
                            scheduled, ready_time, h2d_bytes = self._schedule_transfer(
                                expert_key,
                                prefetch_start_time,
                                'prefetch',
                                cache,
                                inflight,
                                speculative_entries,
                                h2d_state,
                                h2d_bytes,
                                set(pinned_resident_keys),
                            )
                            if not scheduled:
                                funnel_admission_dropped += 1
                                continue

                            funnel_scheduled += 1
                            if best_request.get('is_tail', False):
                                tail_funnel_scheduled += 1
                            prefetch_log[expert_key] = {
                                'predicted_at_token': token_idx,
                                'predicted_at_layer': layer_idx,
                                'was_novel_at_predict': True,
                                'is_tail': best_request.get('is_tail', False),
                                'ready_time': ready_time,
                                'score': best_request.get('score', 1.0),
                                'normalized_score': best_request.get('normalized_score', 1.0),
                                'value_score': best_value_info['value_score'],
                                'benefit': best_value_info['benefit'],
                                'queue_cost': best_value_info['queue_cost'],
                                'saved_stall': best_value_info['saved_stall'],
                            }
                    else:
                        for request in novel_requests:
                            expert_key = request['expert_key']
                            if expert_key in cache or expert_key in inflight:
                                continue
                            admitted, _, _, admission_info = self._prefetch_admission_check(
                                token_idx,
                                layer_idx,
                                request,
                                compute_start_time,
                                prefetch_start_time,
                                cache,
                                inflight,
                                h2d_state,
                            )
                            if not admitted:
                                if self.prefetch_admission == 'deadline':
                                    funnel_deadline_rejected += 1
                                continue

                            scheduled, ready_time, h2d_bytes = self._schedule_transfer(
                                expert_key,
                                prefetch_start_time,
                                'prefetch',
                                cache,
                                inflight,
                                speculative_entries,
                                h2d_state,
                                h2d_bytes,
                                set(pinned_resident_keys),
                            )
                            if not scheduled:
                                funnel_admission_dropped += 1
                                continue

                            funnel_scheduled += 1
                            if request.get('is_tail', False):
                                tail_funnel_scheduled += 1
                            prefetch_log[expert_key] = {
                                'predicted_at_token': token_idx,
                                'predicted_at_layer': layer_idx,
                                'was_novel_at_predict': True,
                                'is_tail': request.get('is_tail', False),
                                'ready_time': ready_time,
                                'score': request.get('score', 1.0),
                                'normalized_score': request.get('normalized_score', 1.0),
                                'value_score': admission_info.get('value_score'),
                                'benefit': admission_info.get('benefit'),
                                'queue_cost': admission_info.get('queue_cost'),
                                'saved_stall': admission_info.get('saved_stall', self._transfer_time_ms()),
                            }

                current_time = compute_start_time + self.layer_compute_time_ms

            latencies.append(current_time - token_start_time)

        # 计算统计指标
        total_requests = hits + misses + prefetch_hits + late_prefetches
        hit_rate = (hits + prefetch_hits) / total_requests if total_requests > 0 else 0
        prefetch_hit_rate = prefetch_hits / total_requests if total_requests > 0 else 0
        late_prefetch_rate = late_prefetches / total_requests if total_requests > 0 else 0
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        avg_latency = np.mean(latencies)
        avg_residual_stall_ms = total_residual_stall_ms / len(self.access_sequence) if self.access_sequence else 0.0

        # 正确计算带宽利用率：基于实际传输时间
        total_time_s = current_time / 1000  # ms -> s
        h2d_bandwidth_used_gbps = (h2d_bytes / (1024**3)) / total_time_s if total_time_s > 0 else 0
        gpu_idle_ratio = gpu_idle_time / (gpu_idle_time + total_compute_time) if (gpu_idle_time + total_compute_time) > 0 else 0
        throughput_tokens_per_sec = len(self.access_sequence) / total_time_s if total_time_s > 0 else 0

        return {
            'device_memory_ratio': device_memory_ratio,
            'prefetch_window': prefetch_window,
            'reset_mode': reset_mode,
            'predictor': self.predictor,
            'cache_capacity': cache_capacity,
            'resident_capacity': resident_capacity,
            'speculative_capacity': speculative_capacity,
            'resident_policy': self.resident_policy,
            'num_pinned_residents': len(pinned_resident_keys),
            'hit_rate': hit_rate,
            'prefetch_hit_rate': prefetch_hit_rate,
            'ready_prefetch_hit_rate': prefetch_hit_rate,
            'p50_latency_ms': p50_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'avg_latency_ms': avg_latency,
            'avg_residual_stall_ms': avg_residual_stall_ms,
            'total_residual_stall_ms': total_residual_stall_ms,
            'h2d_bandwidth_gbps': h2d_bandwidth_used_gbps,
            'gpu_idle_ratio': gpu_idle_ratio,
            'throughput_tokens_per_sec': throughput_tokens_per_sec,
            'total_h2d_bytes': h2d_bytes,
            'hits': hits,
            'misses': misses,
            'prefetch_hits': prefetch_hits,
            'late_prefetches': late_prefetches,
            'late_prefetch_rate': late_prefetch_rate,
            # Funnel metrics: predicted -> novel -> scheduled -> timely -> useful
            'funnel_predicted': funnel_predicted,
            'funnel_novel': funnel_novel,
            'funnel_scheduled': funnel_scheduled,
            'funnel_admission_dropped': funnel_admission_dropped,
            'funnel_deadline_rejected': funnel_deadline_rejected,
            'funnel_value_rejected': funnel_value_rejected,
            'funnel_timely': funnel_timely,
            'funnel_late': funnel_late,
            'funnel_useful': funnel_useful,
            'funnel_useful_stall_saved_ms': funnel_useful_stall_saved_ms,
            'tail_funnel_predicted': tail_funnel_predicted,
            'tail_funnel_novel': tail_funnel_novel,
            'tail_funnel_scheduled': tail_funnel_scheduled,
            'tail_funnel_timely': tail_funnel_timely,
            'tail_funnel_late': tail_funnel_late,
            'tail_funnel_useful': tail_funnel_useful,
            'tail_funnel_useful_stall_saved_ms': tail_funnel_useful_stall_saved_ms,
            'funnel_novel_rate': funnel_novel / max(1, funnel_predicted),
            'funnel_deadline_rejected_rate': funnel_deadline_rejected / max(1, funnel_novel),
            'funnel_value_rejected_rate': funnel_value_rejected / max(1, funnel_novel),
            'funnel_scheduled_rate': funnel_scheduled / max(1, funnel_novel),
            'funnel_timely_rate': funnel_timely / max(1, funnel_scheduled),
            'funnel_late_rate': funnel_late / max(1, funnel_scheduled),
            'funnel_useful_rate': funnel_useful / max(1, funnel_predicted),
            'funnel_useful_given_timely_rate': funnel_useful / max(1, funnel_timely),
            'tail_funnel_novel_rate': tail_funnel_novel / max(1, tail_funnel_predicted),
            'tail_funnel_scheduled_rate': tail_funnel_scheduled / max(1, tail_funnel_novel),
            'tail_funnel_timely_rate': tail_funnel_timely / max(1, tail_funnel_scheduled),
            'tail_funnel_late_rate': tail_funnel_late / max(1, tail_funnel_scheduled),
            'tail_funnel_useful_rate': tail_funnel_useful / max(1, tail_funnel_predicted),
            'tail_funnel_useful_given_timely_rate': tail_funnel_useful / max(1, tail_funnel_timely),
            'per_layer_hits': per_layer_hits,
            'per_layer_misses': per_layer_misses,
            'per_layer_prefetch_hits': per_layer_prefetch_hits,
            'per_layer_late_prefetches': per_layer_late_prefetches,
            'per_layer_residual_stall_ms': per_layer_residual_stall_ms,
            'resident_set': self._serialize_expert_pairs(pinned_resident_keys),
            'per_expert_accesses': self._serialize_expert_metric(per_expert_accesses),
            'per_expert_critical_stall_ms': self._serialize_expert_metric(per_expert_critical_stall_ms),
            'per_expert_demand_misses': self._serialize_expert_metric(per_expert_demand_misses),
            'per_expert_late_prefetches': self._serialize_expert_metric(per_expert_late_prefetches),
            'config': {
                'expert_size_mb': self.expert_size_mb,
                'h2d_bandwidth_gbps': self.h2d_bandwidth_gbps,
                'gpu_compute_time_ms': self.gpu_compute_time_ms,
                'layer_compute_time_ms': self.layer_compute_time_ms,
                'mode': self.mode,
                'predictor': self.predictor,
                'prefetch_admission': self.prefetch_admission,
                'deadline_margin_ms': self.deadline_margin_ms,
                'value_cost_scale': self.value_cost_scale,
                'cache_layout': self.cache_layout,
                'resident_ratio': self.resident_ratio,
                'resident_policy': self.resident_policy,
                'resident_profile_ratio': self.resident_profile_ratio,
                'resident_depth_power': self.resident_depth_power,
            }
        }

    def run_sweep(self, reset_mode='shared', memory_ratios=None, prefetch_windows=None):
        """
        运行二维参数扫描

        reset_mode: 'per_sequence' (每序列重置cache) 或 'shared' (跨序列共享)
        """
        # 参数范围
        memory_ratios = memory_ratios or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        prefetch_windows = prefetch_windows or [0, 1, 2, 3, 4, 5, 6, 8, 10]

        print(
            f"Running parameter sweep (mode={self.mode}, predictor={self.predictor}, "
            f"reset={reset_mode}, admission={self.prefetch_admission}, "
            f"layout={self.cache_layout}, resident_policy={self.resident_policy}, "
            f"resident_profile_ratio={self.resident_profile_ratio}, "
            f"resident_depth_power={self.resident_depth_power}, "
            f"expert_size_mb={self.expert_size_mb})..."
        )
        print(f"Memory ratios: {memory_ratios}")
        print(f"Prefetch windows: {prefetch_windows}")
        print(f"Total configurations: {len(memory_ratios) * len(prefetch_windows)}")

        results = []
        for mem_ratio in memory_ratios:
            for prefetch_win in prefetch_windows:
                print(f"  Simulating: memory_ratio={mem_ratio:.1f}, prefetch_window={prefetch_win}")
                result = self.simulate_with_config(mem_ratio, prefetch_win, reset_mode=reset_mode)
                results.append(result)

        # 保存结果 (文件名包含配置)
        output_file = self._result_file(reset_mode)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_file}")
        return results

    def plot_heatmaps(self, results):
        """绘制热力图"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 提取数据
        memory_ratios = sorted(set(r['device_memory_ratio'] for r in results))
        prefetch_windows = sorted(set(r['prefetch_window'] for r in results))

        metrics = [
            'p95_latency_ms',
            'throughput_tokens_per_sec',
            'avg_residual_stall_ms',
            'hit_rate',
            'gpu_idle_ratio',
            'ready_prefetch_hit_rate',
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            # 构建热力图数据
            data = np.zeros((len(prefetch_windows), len(memory_ratios)))
            for r in results:
                i = prefetch_windows.index(r['prefetch_window'])
                j = memory_ratios.index(r['device_memory_ratio'])
                data[i, j] = r[metric]

            # 绘制热力图
            ax = axes[idx]
            sns.heatmap(data,
                       xticklabels=[f"{m:.1f}" for m in memory_ratios],
                       yticklabels=prefetch_windows,
                       annot=True,
                       fmt='.2f',
                       cmap='RdYlGn_r' if 'latency' in metric or 'idle' in metric or 'stall' in metric else 'RdYlGn',
                       ax=ax,
                       cbar_kws={'label': metric})
            ax.set_xlabel('Device Memory Ratio')
            ax.set_ylabel('Prefetch Window')
            ax.set_title(f'{metric.replace("_", " ").title()}')

        plt.tight_layout()
        output_file = self.output_dir / "../figures/insight_4_heatmaps.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Heatmaps saved to {output_file}")
        plt.close()

    def find_phase_transitions(self, results):
        """找到性能相变点"""
        print("\n" + "="*80)
        print("Phase Transition Analysis")
        print("="*80)

        # 按 prefetch_window 分组，找到 memory_ratio 的相变点
        prefetch_windows = sorted(set(r['prefetch_window'] for r in results))

        for prefetch_win in prefetch_windows:
            subset = [r for r in results if r['prefetch_window'] == prefetch_win]
            subset = sorted(subset, key=lambda x: x['device_memory_ratio'])

            throughputs = [r['throughput_tokens_per_sec'] for r in subset]
            stalls = [r['avg_residual_stall_ms'] for r in subset]

            throughput_gradients = np.diff(throughputs)
            stall_drops = np.diff(stalls)

            print(f"\nPrefetch Window = {prefetch_win}:")
            if len(throughput_gradients) > 0:
                max_thr_idx = np.argmax(throughput_gradients)
                print(
                    "  Largest throughput jump: "
                    f"{throughputs[max_thr_idx]:.3f} -> {throughputs[max_thr_idx + 1]:.3f} "
                    f"at memory_ratio {subset[max_thr_idx]['device_memory_ratio']:.2f} -> "
                    f"{subset[max_thr_idx + 1]['device_memory_ratio']:.2f}"
                )
            if len(stall_drops) > 0:
                max_stall_idx = np.argmin(stall_drops)
                print(
                    "  Largest residual-stall drop: "
                    f"{stalls[max_stall_idx]:.3f} -> {stalls[max_stall_idx + 1]:.3f} "
                    f"at memory_ratio {subset[max_stall_idx]['device_memory_ratio']:.2f} -> "
                    f"{subset[max_stall_idx + 1]['device_memory_ratio']:.2f}"
                )

    def print_summary(self, results):
        """打印摘要"""
        print("\n" + "="*80)
        print("System Bottleneck Analysis Summary")
        print("="*80)

        # 找到最优配置
        best_throughput = max(results, key=lambda x: x['throughput_tokens_per_sec'])
        best_latency = min(results, key=lambda x: x['p95_latency_ms'])
        best_hit_rate = max(results, key=lambda x: x['hit_rate'])
        best_residual_stall = min(results, key=lambda x: x['avg_residual_stall_ms'])

        print("\nBest Configurations:")
        print(f"\n1. Highest Throughput:")
        print(f"   Memory Ratio: {best_throughput['device_memory_ratio']:.2f}")
        print(f"   Prefetch Window: {best_throughput['prefetch_window']}")
        print(f"   Throughput: {best_throughput['throughput_tokens_per_sec']:.2f} tokens/sec")
        print(f"   P95 Latency: {best_throughput['p95_latency_ms']:.2f}ms")
        print(f"   Avg Residual Stall: {best_throughput['avg_residual_stall_ms']:.2f}ms")

        print(f"\n2. Lowest P95 Latency:")
        print(f"   Memory Ratio: {best_latency['device_memory_ratio']:.2f}")
        print(f"   Prefetch Window: {best_latency['prefetch_window']}")
        print(f"   P95 Latency: {best_latency['p95_latency_ms']:.2f}ms")
        print(f"   Hit Rate: {best_latency['hit_rate']:.3f}")

        print(f"\n3. Lowest Residual Stall:")
        print(f"   Memory Ratio: {best_residual_stall['device_memory_ratio']:.2f}")
        print(f"   Prefetch Window: {best_residual_stall['prefetch_window']}")
        print(f"   Avg Residual Stall: {best_residual_stall['avg_residual_stall_ms']:.2f}ms")
        print(f"   Throughput: {best_residual_stall['throughput_tokens_per_sec']:.2f} tokens/sec")

        print(f"\n4. Highest Hit Rate:")
        print(f"   Memory Ratio: {best_hit_rate['device_memory_ratio']:.2f}")
        print(f"   Prefetch Window: {best_hit_rate['prefetch_window']}")
        print(f"   Hit Rate: {best_hit_rate['hit_rate']:.3f}")
        print(f"   P95 Latency: {best_hit_rate['p95_latency_ms']:.2f}ms")


if __name__ == "__main__":
    import argparse

    def parse_float_list(text):
        return [float(item.strip()) for item in text.split(",") if item.strip()]

    def parse_int_list(text):
        return [int(item.strip()) for item in text.split(",") if item.strip()]

    parser = argparse.ArgumentParser(description='Insight 4: System Bottleneck Analysis')
    parser.add_argument('--mode', type=str, default='causal', choices=['causal', 'oracle'],
                        help='Prediction mode: causal (use history) or oracle (use future)')
    parser.add_argument('--predictor', type=str, default='history_freq',
                        choices=['history_freq', 'pl_ctr', 'utility_freq'],
                        help='Predictor used in causal mode')
    parser.add_argument('--reset', type=str, default='shared', choices=['shared', 'per_sequence'],
                        help='Cache reset mode: shared (across sequences) or per_sequence')
    parser.add_argument('--expert-size-mb', type=float, default=50.0,
                        help='Expert size in MiB-equivalent units used by the simulator')
    parser.add_argument('--h2d-bandwidth-gbps', type=float, default=16.0,
                        help='Effective host-to-device bandwidth in GiB/s')
    parser.add_argument('--gpu-compute-time-ms', type=float, default=2.0,
                        help='Per-token compute time in ms')
    parser.add_argument('--prefetch-admission', type=str, default='none',
                        choices=['none', 'deadline', 'value'],
                        help='Prefetch admission policy')
    parser.add_argument('--deadline-margin-ms', type=float, default=0.0,
                        help='Safety margin for deadline admission; prefetch must arrive this much earlier than due time')
    parser.add_argument('--value-cost-scale', type=float, default=1.0,
                        help='Scale factor for queue-cost in value-aware admission')
    parser.add_argument('--cache-layout', type=str, default='single',
                        choices=['single', 'two_pool'],
                        help='Cache layout: shared single pool or resident/speculative two-pool')
    parser.add_argument('--resident-ratio', type=float, default=0.5,
                        help='Resident-pool ratio under two_pool layout')
    parser.add_argument('--resident-policy', type=str, default='none',
                        choices=['none', 'oracle_freq', 'profile_freq', 'profile_depth_freq', 'profile_miss_stall'],
                        help='How to seed the resident pool under two_pool')
    parser.add_argument('--resident-profile-ratio', type=float, default=0.1,
                        help='Prefix ratio used by profile_freq to build resident-pool statistics')
    parser.add_argument('--resident-depth-power', type=float, default=1.0,
                        help='Depth exponent used by profile_depth_freq')
    parser.add_argument('--output-tag', type=str, default='',
                        help='Optional suffix appended to the result filename')
    parser.add_argument('--state-file', type=str, default="../states/Qwen1.5-MoE-A2.7B-Chat~lmsys-chat-1m~8.pkl",
                        help='Path to routed trace state file')
    parser.add_argument('--memory-ratios', type=parse_float_list, default=None,
                        help='Optional comma-separated memory ratios, e.g. 0.02,0.05,0.1')
    parser.add_argument('--prefetch-windows', type=parse_int_list, default=None,
                        help='Optional comma-separated prefetch windows, e.g. 0,1,2,4')
    args = parser.parse_args()

    analyzer = SystemBottleneckAnalyzer(
        args.state_file,
        mode=args.mode,
        predictor=args.predictor,
        expert_size_mb=args.expert_size_mb,
        h2d_bandwidth_gbps=args.h2d_bandwidth_gbps,
        gpu_compute_time_ms=args.gpu_compute_time_ms,
        prefetch_admission=args.prefetch_admission,
        deadline_margin_ms=args.deadline_margin_ms,
        value_cost_scale=args.value_cost_scale,
        cache_layout=args.cache_layout,
        resident_ratio=args.resident_ratio,
        resident_policy=args.resident_policy,
        resident_profile_ratio=args.resident_profile_ratio,
        resident_depth_power=args.resident_depth_power,
        output_tag=args.output_tag,
    )
    results = analyzer.run_sweep(
        reset_mode=args.reset,
        memory_ratios=args.memory_ratios,
        prefetch_windows=args.prefetch_windows,
    )
    analyzer.print_summary(results)
    analyzer.find_phase_transitions(results)
    # analyzer.plot_heatmaps(results)  # 可选：绘图
