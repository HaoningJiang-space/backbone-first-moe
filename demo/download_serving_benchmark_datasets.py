import argparse
import json
import os
import random
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset


def _gsm8k_prompt(example):
    return example["question"].strip()


def _mmlu_prompt(example):
    choices = example.get("choices") or []
    rendered = [f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(choices)]
    body = "\n".join(rendered)
    return f"{example['question'].strip()}\n{body}".strip()


def _bbh_prompt(example):
    return example.get("input", "").strip()


def _longbench_prompt(example):
    parts = []
    for key in ("context", "article", "passage", "input"):
        value = example.get(key)
        if value:
            parts.append(str(value).strip())
    prompt = "\n\n".join(part for part in parts if part)
    return prompt[:20000].strip()


BENCHMARK_SPECS = {
    "gsm8k": {
        "dataset_path": "gsm8k",
        "config": "main",
        "split": "test",
        "extract_prompt": _gsm8k_prompt,
    },
    "mmlu": {
        "dataset_path": "cais/mmlu",
        "config": "all",
        "split": "test",
        "extract_prompt": _mmlu_prompt,
    },
    "bbh": {
        "dataset_path": "lukaemon/bbh",
        "config": None,
        "split": "test",
        "extract_prompt": _bbh_prompt,
    },
}

LONGBENCH_DEFAULT_CONFIGS = [
    "hotpotqa",
    "2wikimqa",
    "musique",
    "qasper",
    "multifieldqa_en",
]


def _write_prompt_json(path: Path, records):
    payload = [{"prompt": item["prompt"], **{k: v for k, v in item.items() if k != "prompt"}} for item in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _sample_records(dataset, extractor, sample_size: int, seed: int, *, extra_fields=None):
    extra_fields = extra_fields or ()
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    records = []
    for idx in indices:
        example = dataset[idx]
        prompt = extractor(example)
        if not prompt:
            continue
        record = {
            "prompt": prompt,
            "source_index": int(idx),
        }
        for field in extra_fields:
            if field in example:
                record[field] = example[field]
        records.append(record)
        if len(records) >= sample_size:
            break
    return records


def _download_standard_benchmark(name: str, spec: dict, *, cache_dir: str | None, output_dir: Path, sample_size: int, seed: int):
    dataset = load_dataset(
        spec["dataset_path"],
        spec["config"],
        split=spec["split"],
        cache_dir=cache_dir,
    )
    records = _sample_records(
        dataset,
        spec["extract_prompt"],
        sample_size,
        seed,
        extra_fields=("question", "subject", "task_name"),
    )
    output_path = output_dir / f"{name}~eval_prompts.json"
    _write_prompt_json(output_path, records)
    return {
        "name": name,
        "dataset_path": spec["dataset_path"],
        "config": spec["config"],
        "split": spec["split"],
        "sample_size": len(records),
        "output_file": str(output_path),
    }


def _download_longbench(*, cache_dir: str | None, output_dir: Path, sample_size: int, seed: int, configs=None):
    available = set(get_dataset_config_names("THUDM/LongBench", cache_dir=cache_dir))
    selected = [cfg for cfg in (configs or LONGBENCH_DEFAULT_CONFIGS) if cfg in available]
    if not selected:
        raise RuntimeError("No requested LongBench configs are available")

    per_config = max(1, sample_size // len(selected))
    combined = []
    manifest = []
    for cfg in selected:
        dataset = load_dataset(
            "THUDM/LongBench",
            cfg,
            split="test",
            cache_dir=cache_dir,
        )
        records = _sample_records(
            dataset,
            _longbench_prompt,
            per_config,
            seed,
            extra_fields=("input",),
        )
        for item in records:
            item["longbench_config"] = cfg
        output_path = output_dir / f"longbench_{cfg}~eval_prompts.json"
        _write_prompt_json(output_path, records)
        manifest.append(
            {
                "name": f"longbench_{cfg}",
                "dataset_path": "THUDM/LongBench",
                "config": cfg,
                "split": "test",
                "sample_size": len(records),
                "output_file": str(output_path),
            }
        )
        combined.extend(records)

    random.Random(seed).shuffle(combined)
    combined = combined[:sample_size]
    combined_path = output_dir / "longbench_mixed~eval_prompts.json"
    _write_prompt_json(combined_path, combined)
    manifest.append(
        {
            "name": "longbench_mixed",
            "dataset_path": "THUDM/LongBench",
            "config": selected,
            "split": "test",
            "sample_size": len(combined),
            "output_file": str(combined_path),
        }
    )
    return manifest


def _build_mixed_workload(output_dir: Path, manifest: list[dict], seed: int, per_source: int):
    merged = []
    for item in manifest:
        if item["name"] == "longbench_mixed":
            continue
        path = Path(item["output_file"])
        payload = json.loads(path.read_text())
        random.Random(seed).shuffle(payload)
        take = payload[:per_source]
        for entry in take:
            merged.append(
                {
                    "prompt": entry["prompt"],
                    "source_dataset": item["name"],
                }
            )
    random.Random(seed).shuffle(merged)
    out = output_dir / "serving_benchmark_mixed~eval_prompts.json"
    _write_prompt_json(out, merged)
    return {
        "name": "serving_benchmark_mixed",
        "sample_size": len(merged),
        "output_file": str(out),
    }


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets and export serving prompt JSON files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets",
        type=str,
        default="gsm8k,mmlu,bbh,longbench",
        help="Comma-separated benchmark set names",
    )
    parser.add_argument("--mixed-per-source", type=int, default=32)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = [item.strip() for item in args.datasets.split(",") if item.strip()]
    manifest = []
    errors = []

    for name in requested:
        try:
            if name == "longbench":
                manifest.extend(
                    _download_longbench(
                        cache_dir=args.cache_dir,
                        output_dir=output_dir,
                        sample_size=args.sample_size,
                        seed=args.seed,
                    )
                )
            else:
                spec = BENCHMARK_SPECS[name]
                manifest.append(
                    _download_standard_benchmark(
                        name,
                        spec,
                        cache_dir=args.cache_dir,
                        output_dir=output_dir,
                        sample_size=args.sample_size,
                        seed=args.seed,
                    )
                )
        except Exception as exc:  # pragma: no cover - runtime-facing
            errors.append({"dataset": name, "error": str(exc)})

    if manifest:
        manifest.append(
            _build_mixed_workload(
                output_dir,
                manifest,
                seed=args.seed,
                per_source=args.mixed_per_source,
            )
        )

    summary = {
        "requested_datasets": requested,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "outputs": manifest,
        "errors": errors,
    }
    summary_path = output_dir / "benchmark_download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
