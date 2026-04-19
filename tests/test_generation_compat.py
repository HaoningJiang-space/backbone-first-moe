from transformers.generation import GenerationMixin

from finemoe.models.modeling_olmoe.modeling_olmoe import OlmoeForCausalLM
from finemoe.models.modeling_qwen.modeling_qwen2_moe import Qwen2MoeForCausalLM


def test_qwen_generation_mixin_present():
    assert issubclass(Qwen2MoeForCausalLM, GenerationMixin)


def test_olmoe_generation_mixin_present():
    assert issubclass(OlmoeForCausalLM, GenerationMixin)
