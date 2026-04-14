import argparse
import json
from pathlib import Path


WORKLOAD_SPECS = {
    "technical_8": {
        "description": "Technical / code / systems-oriented prompts curated from the LMSYS prompt pool.",
        "indices": [16, 17, 28, 32, 38, 44, 49, 63],
    },
    "creative_roleplay_8": {
        "description": "Creative writing / roleplay prompts curated from the LMSYS prompt pool.",
        "indices": [8, 15, 34, 39, 41, 43, 56, 60],
    },
}


def load_prompts(path):
    with open(path) as f:
        data = json.load(f)
    prompts = []
    for item in data:
        if isinstance(item, dict):
            prompts.append(item["prompt"])
        else:
            prompts.append(str(item))
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Build curated workload prompt sets from the LMSYS prompt pool.")
    parser.add_argument(
        "--source-prompt-file",
        type=str,
        default="./states/lmsys-chat-1m~eval_prompts.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./states",
    )
    args = parser.parse_args()

    prompts = load_prompts(args.source_prompt_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for workload_name, spec in WORKLOAD_SPECS.items():
        records = []
        for idx in spec["indices"]:
            records.append({
                "source_index": idx,
                "prompt": prompts[idx],
            })

        output_path = output_dir / f"{workload_name}~eval_prompts.json"
        output_path.write_text(json.dumps(records, indent=2))

        print(f"Saved {workload_name} to {output_path}")
        print(f"Description: {spec['description']}")
        for item in records:
            snippet = item["prompt"][:120].replace("\n", " ")
            print(f"  idx={item['source_index']}: {snippet}")
        print("")


if __name__ == "__main__":
    main()
