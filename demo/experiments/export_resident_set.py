import argparse
import json
from pathlib import Path

from insight_4_system_bottleneck import SystemBottleneckAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Export a runtime resident expert JSON from the simulator resident policy."
    )
    parser.add_argument("--state-file", type=str, required=True)
    parser.add_argument("--device-memory-ratio", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./experiments/results"))
    parser.add_argument("--resident-ratio", type=float, default=0.9)
    parser.add_argument("--resident-policy", type=str, default="profile_freq")
    parser.add_argument("--resident-profile-ratio", type=float, default=0.2)
    parser.add_argument("--resident-depth-power", type=float, default=1.0)
    parser.add_argument("--expert-size-mb", type=float, default=17.2)
    parser.add_argument("--reset-mode", type=str, default="shared", choices=["shared", "per_sequence"])
    args = parser.parse_args()

    analyzer = SystemBottleneckAnalyzer(
        state_file=args.state_file,
        mode="causal",
        predictor="history_freq",
        output_dir=str(args.output_dir),
        expert_size_mb=args.expert_size_mb,
        cache_layout="two_pool",
        resident_ratio=args.resident_ratio,
        resident_policy=args.resident_policy,
        resident_profile_ratio=args.resident_profile_ratio,
        resident_depth_power=args.resident_depth_power,
    )

    resident_info = analyzer.get_resident_set(
        device_memory_ratio=args.device_memory_ratio,
        reset_mode=args.reset_mode,
    )
    payload = {
        "state_file": args.state_file,
        "device_memory_ratio": float(args.device_memory_ratio),
        "reset_mode": args.reset_mode,
        "resident_ratio": float(args.resident_ratio),
        "resident_policy": args.resident_policy,
        "resident_profile_ratio": float(args.resident_profile_ratio),
        "resident_depth_power": float(args.resident_depth_power),
        "cache_capacity": int(resident_info["cache_capacity"]),
        "resident_capacity": int(resident_info["resident_capacity"]),
        "speculative_capacity": int(resident_info["speculative_capacity"]),
        "resident_set": resident_info["resident_set"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(
        f"Saved resident set to {args.output} "
        f"(count={len(payload['resident_set'])}, mem={args.device_memory_ratio:.2f})"
    )


if __name__ == "__main__":
    main()
