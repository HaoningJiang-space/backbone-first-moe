from transformers import MixtralConfig

from demo.prepare_custom_data import resolve_moe_name, resolve_trace_prefetch_distance
from finemoe.models.modeling_qwen.configuration_qwen2_moe import Qwen2MoeConfig


def test_resolve_moe_name_uses_cli_model_path():
    assert resolve_moe_name("/data/ziheng/models/OLMoE-1B-7B-0924") == "OLMoE-1B-7B-0924"


def test_resolve_moe_name_strips_trailing_slash():
    assert resolve_moe_name("/data/models/Qwen1.5-MoE-A2.7B-Chat/") == "Qwen1.5-MoE-A2.7B-Chat"


def test_resolve_trace_prefetch_distance_for_packed_moe():
    assert resolve_trace_prefetch_distance(MixtralConfig()) == 0


def test_resolve_trace_prefetch_distance_keeps_modulelist_default():
    assert resolve_trace_prefetch_distance(Qwen2MoeConfig()) > 0
