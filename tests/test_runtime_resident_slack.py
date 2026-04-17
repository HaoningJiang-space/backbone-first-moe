from pathlib import Path

from finemoe.backbone.runtime_eval import RuntimeEvalConfig, derive_resident_slack_experts


class StubTokenizer:
    def __call__(self, prompts, truncation, padding, max_length, return_tensors):
        return {"input_ids": [[0] * min(len(prompt), max_length) for prompt in prompts]}


def test_derive_resident_slack_uses_prefill_frontier_over_speculative_capacity():
    cfg = RuntimeEvalConfig(
        model_path="dummy",
        prompt_file=Path("dummy.json"),
        output=Path("dummy.out"),
        offload_path="dummy",
        device_memory_ratio=0.1,
        resident_expert_ids_file="resident.json",
        prefetch_distance=0,
        max_length=256,
        resident_slack_experts=-1,
    )
    prompts = ["a" * 32, "b" * 8]
    resident_metadata = {"speculative_capacity": 48}
    slack = derive_resident_slack_experts(
        cfg,
        prompts,
        StubTokenizer(),
        resident_metadata=resident_metadata,
        top_k=4,
    )
    assert slack == 128


def test_derive_resident_slack_respects_explicit_override():
    cfg = RuntimeEvalConfig(
        model_path="dummy",
        prompt_file=Path("dummy.json"),
        output=Path("dummy.out"),
        offload_path="dummy",
        device_memory_ratio=0.1,
        resident_expert_ids_file="resident.json",
        prefetch_distance=0,
        max_length=256,
        resident_slack_experts=96,
    )
    slack = derive_resident_slack_experts(
        cfg,
        ["a" * 32],
        StubTokenizer(),
        resident_metadata={"speculative_capacity": 48},
        top_k=4,
    )
    assert slack == 96
