import torch
from transformers.generation import GenerationMixin

from finemoe.models.modeling_olmoe.modeling_olmoe import _get_cache_length as olmoe_get_cache_length
from finemoe.models.modeling_olmoe.modeling_olmoe import OlmoeForCausalLM
from finemoe.models.modeling_qwen.modeling_qwen2_moe import _get_cache_length as qwen_get_cache_length
from finemoe.models.modeling_qwen.modeling_qwen2_moe import Qwen2MoeForCausalLM


def test_qwen_generation_mixin_present():
    assert issubclass(Qwen2MoeForCausalLM, GenerationMixin)


def test_olmoe_generation_mixin_present():
    assert issubclass(OlmoeForCausalLM, GenerationMixin)


def test_qwen_prepare_inputs_handles_missing_cache_position():
    model = object.__new__(Qwen2MoeForCausalLM)
    inputs = torch.tensor([[1, 2, 3]])
    prepared = model.prepare_inputs_for_generation(
        input_ids=inputs,
        past_key_values=object(),
        attention_mask=torch.ones_like(inputs),
        cache_position=None,
    )
    assert prepared["input_ids"].tolist() == [[3]]


def test_olmoe_prepare_inputs_handles_missing_cache_position():
    model = object.__new__(OlmoeForCausalLM)
    inputs = torch.tensor([[4, 5, 6]])
    prepared = model.prepare_inputs_for_generation(
        input_ids=inputs,
        past_key_values=object(),
        attention_mask=torch.ones_like(inputs),
        cache_position=None,
    )
    assert prepared["input_ids"].tolist() == [[6]]


class _LegacyCache:
    def get_usable_length(self, kv_seq_len, layer_idx):
        assert kv_seq_len == 7
        assert layer_idx == 3
        return 5


class _ModernCache:
    def get_seq_length(self, layer_idx=0):
        assert layer_idx == 3
        return 4


def test_qwen_cache_length_helper_supports_legacy_and_modern_cache():
    assert qwen_get_cache_length(_LegacyCache(), 3, 7) == 5
    assert qwen_get_cache_length(_ModernCache(), 3, 7) == 4


def test_olmoe_cache_length_helper_supports_legacy_and_modern_cache():
    assert olmoe_get_cache_length(_LegacyCache(), 3, 7) == 5
    assert olmoe_get_cache_length(_ModernCache(), 3, 7) == 4
