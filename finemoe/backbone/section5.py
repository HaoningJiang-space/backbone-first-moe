import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
DEFAULT_OFFLOAD_PATH = "/data/jiangjunmin/jhn/FineMoE-EuroSys26/demo/offloads/Qwen1.5-MoE-A2.7B-Chat"
DEFAULT_PROMPT_FILE = "/data/jiangjunmin/jhn/FineMoE-EuroSys26/demo/states/lmsys-chat-1m-shuffled64-seed42~eval_prompts.json"
DEFAULT_STORE_PREFIX = "/data/jiangjunmin/jhn/FineMoE-EuroSys26/demo/states/Qwen1.5-MoE-A2.7B-Chat~lmsys-chat-1m-shuffled64-seed42"
DEFAULT_RESIDENT_DIR = "/data/jiangjunmin/jhn/FineMoE-EuroSys26/demo/experiments/experiments/results"
DEFAULT_OUTPUT_DIR = "/data/jiangjunmin/jhn/FineMoE-EuroSys26/demo/experiments/experiments/results/section5_sweep"
DEFAULT_PYTHON = sys.executable
DEFAULT_EVAL_SCRIPT = str(REPO_ROOT / "demo" / "experiments" / "run_runtime_eval.py")


@dataclass
class RuntimeSweepArgs:
    model_path: str
    offload_path: str
    prompt_file: Path
    store_prefix: str
    resident_dir: str
    output_dir: str
    python_bin: str
    eval_script: str
    memory_ratios: str
    prefetch_distance: int
    store_capacity: int
    device: str
    batch_size: int
    num_prompts: int
    seed: int
    max_length: int
    max_new_tokens: int
    min_new_tokens: int


def resident_file_for_mem(resident_dir, mem_ratio):
    tag = f"{mem_ratio:.2f}".replace(".", "p")
    return str(Path(resident_dir) / f"resident_set_mem{tag}_profile_freq.json")


def build_runtime_sweep_configs(args):
    configs = []
    memory_ratios = [float(x) for x in args.memory_ratios.split(",")]

    for mem in memory_ratios:
        mem_tag = str(mem).replace(".", "p")
        common = [
            "--model-path", args.model_path,
            "--prompt-file", str(args.prompt_file),
            "--offload-path", args.offload_path,
            "--device-memory-ratio", str(mem),
            "--device", args.device,
            "--batch-size", str(args.batch_size),
            "--num-prompts", str(args.num_prompts),
            "--seed", str(args.seed),
            "--max-length", str(args.max_length),
            "--max-new-tokens", str(args.max_new_tokens),
            "--min-new-tokens", str(args.min_new_tokens),
            "--store-capacity", str(args.store_capacity),
        ]

        configs.append((
            f"A_demand_mem{mem_tag}",
            common + [
                "--prefetch-distance", "0",
                "--store-prefix", args.store_prefix,
                "--resident-expert-ids-file", "",
                "--eval-mode", "offline",
                "--tag", f"A_demand_mem{mem_tag}",
            ],
        ))

        configs.append((
            f"B_lru_prefetch_mem{mem_tag}",
            common + [
                "--prefetch-distance", str(args.prefetch_distance),
                "--store-prefix", args.store_prefix,
                "--resident-expert-ids-file", "",
                "--eval-mode", "offline",
                "--tag", f"B_lru_prefetch_mem{mem_tag}",
            ],
        ))

        res_file = resident_file_for_mem(args.resident_dir, mem)
        if not Path(res_file).exists():
            print(f"WARNING: resident file not found, skipping configs C/D for mem={mem}: {res_file}")
            continue

        configs.append((
            f"C_backbone_only_mem{mem_tag}",
            common + [
                "--prefetch-distance", "0",
                "--store-prefix", args.store_prefix,
                "--resident-expert-ids-file", res_file,
                "--eval-mode", "offline",
                "--tag", f"C_backbone_only_mem{mem_tag}",
            ],
        ))

        configs.append((
            f"D_backbone_mem{mem_tag}",
            common + [
                "--prefetch-distance", str(args.prefetch_distance),
                "--store-prefix", args.store_prefix,
                "--resident-expert-ids-file", res_file,
                "--eval-mode", "offline",
                "--tag", f"D_backbone_mem{mem_tag}",
            ],
        ))

    return configs


def run_runtime_config(python_bin, eval_script, label, cli_args, output_dir):
    output_path = Path(output_dir) / f"{label}.json"
    cmd = [python_bin, eval_script] + cli_args + ["--output", str(output_path)]
    print(f"\n{'=' * 60}")
    print(f"Running: {label}")
    print(f"  output: {output_path}")
    print(f"{'=' * 60}", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent)
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    if result.returncode != 0:
        print(f"ERROR: {label} failed with return code {result.returncode}")
        return None
    if not output_path.exists():
        print(f"ERROR: output file not created for {label}")
        return None
    with open(output_path) as f:
        return json.load(f)


def format_runtime_summary(all_results):
    if not all_results:
        return "No results to summarize."

    lines = []
    lines.append("=" * 100)
    lines.append(f"{'Section 5 Real-Hardware Evaluation Summary':^100}")
    lines.append("=" * 100)
    lines.append(
        f"{'Config':<30} {'mem':>5} {'gen tok/s':>10} {'e2e tok/s':>10} "
        f"{'elapsed(s)':>10} {'peak MB':>10} {'resident':>8}"
    )
    lines.append("-" * 100)
    for label, data in all_results:
        mem = data.get("device_memory_ratio", 0)
        gen_tps = data.get("generated_tokens_per_sec", 0)
        e2e_tps = data.get("end_to_end_tokens_per_sec", 0)
        elapsed = data.get("total_elapsed_sec", 0)
        peak = data.get("peak_memory_mb")
        res_count = data.get("resident_count", 0)
        peak_str = f"{peak:.0f}" if peak is not None else "N/A"
        lines.append(
            f"{label:<30} {mem:>5.2f} {gen_tps:>10.2f} {e2e_tps:>10.2f} "
            f"{elapsed:>10.2f} {peak_str:>10} {res_count:>8}"
        )
    lines.append("=" * 100)
    return "\n".join(lines)
