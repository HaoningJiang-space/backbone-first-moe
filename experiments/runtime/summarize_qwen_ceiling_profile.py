#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYS = [
    "generated_tokens_per_sec",
    "total_elapsed_sec",
    "module_begin_wall_time_sec",
    "module_end_wall_time_sec",
    "tail_group_begin_wall_time_sec",
    "tail_group_end_wall_time_sec",
    "tail_group_compute_wall_time_sec",
    "modulelist_expert_compute_wall_time_sec",
    "modulelist_resident_compute_wall_time_sec",
    "modulelist_demand_compute_wall_time_sec",
    "resident_fastpath_module_skips",
]


def load_json(path: Path):
    return json.loads(path.read_text())


def extract_profile(window_payload: dict):
    runtime_profile = window_payload.get("runtime_profile", {})
    flat = {
        "generated_tokens_per_sec": window_payload.get("generated_tokens_per_sec"),
        "total_elapsed_sec": window_payload.get("total_elapsed_sec"),
    }
    for key in KEYS[2:]:
        flat[key] = runtime_profile.get(key)
    return flat


def delta(a, b):
    if a is None or b is None:
        return None
    return b - a


def main():
    parser = argparse.ArgumentParser(description="Summarize Qwen ceiling mode profiles.")
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    payload = load_json(args.summary_json)
    rows = {}
    for mode in ("control", "no_control", "no_tail_wait"):
        mode_payload = payload.get(mode)
        if not mode_payload:
            continue
        window = mode_payload.get("window")
        if not window:
            continue
        rows[mode] = {
            "A": extract_profile(window["A"]),
            "C": extract_profile(window["C"]),
            "gain_percent": window["gain_percent"],
            "num_measured_slices": mode_payload.get("num_measured_slices"),
        }

    report = {"summary_json": str(args.summary_json), "modes": rows, "deltas": {}}
    for mode, row in rows.items():
        report["deltas"][mode] = {
            key: delta(row["A"].get(key), row["C"].get(key))
            for key in KEYS
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
