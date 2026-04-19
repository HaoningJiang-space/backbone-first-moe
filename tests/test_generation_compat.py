import torch
from transformers.generation import GenerationMixin

from finemoe.models.modeling_olmoe.modeling_olmoe import OlmoeForCausalLM
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
