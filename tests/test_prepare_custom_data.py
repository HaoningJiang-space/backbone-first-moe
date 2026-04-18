from demo.prepare_custom_data import resolve_moe_name


def test_resolve_moe_name_uses_cli_model_path():
    assert resolve_moe_name("/data/ziheng/models/OLMoE-1B-7B-0924") == "OLMoE-1B-7B-0924"


def test_resolve_moe_name_strips_trailing_slash():
    assert resolve_moe_name("/data/models/Qwen1.5-MoE-A2.7B-Chat/") == "Qwen1.5-MoE-A2.7B-Chat"
