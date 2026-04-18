import unittest
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "runtime" / "section5_sweep.py"
SPEC = importlib.util.spec_from_file_location("section5_sweep", MODULE_PATH)
SECTION5_SWEEP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SECTION5_SWEEP)

RuntimeSweepArgs = SECTION5_SWEEP.RuntimeSweepArgs
build_runtime_sweep_configs = SECTION5_SWEEP.build_runtime_sweep_configs
format_runtime_summary = SECTION5_SWEEP.format_runtime_summary
resident_file_for_mem = SECTION5_SWEEP.resident_file_for_mem


class Section5HelpersTest(unittest.TestCase):
    def test_resident_file_for_mem(self):
        path = resident_file_for_mem("/tmp/results", 0.10)
        self.assertTrue(path.endswith("resident_set_mem0p10_profile_freq.json"))

    def test_format_runtime_summary(self):
        text = format_runtime_summary(
            [
                (
                    "A_demand_mem0p10",
                    {
                        "device_memory_ratio": 0.10,
                        "generated_tokens_per_sec": 1.23,
                        "end_to_end_tokens_per_sec": 4.56,
                        "total_elapsed_sec": 10.0,
                        "peak_memory_mb": 2048.0,
                        "resident_count": 0,
                    },
                )
            ]
        )
        self.assertIn("Section 5 Real-Hardware Evaluation Summary", text)
        self.assertIn("A_demand_mem0p10", text)
        self.assertIn("4.56", text)

    def test_build_runtime_sweep_configs_without_resident_files(self):
        args = RuntimeSweepArgs(
            model_path="model",
            offload_path="/tmp/offload",
            prompt_file="prompts.json",
            store_prefix="/tmp/store",
            resident_dir="/tmp/nonexistent",
            output_dir="/tmp/output",
            python_bin="python",
            eval_script="run_runtime_eval.py",
            memory_ratios="0.10",
            prefetch_distance=6,
            store_capacity=1000,
            device="cuda:0",
            batch_size=1,
            num_prompts=8,
            seed=42,
            max_length=256,
            max_new_tokens=64,
            min_new_tokens=1,
        )
        configs = build_runtime_sweep_configs(args)
        labels = [label for label, _ in configs]
        self.assertEqual(labels, ["A_demand_mem0p1", "B_lru_prefetch_mem0p1"])


if __name__ == "__main__":
    unittest.main()
