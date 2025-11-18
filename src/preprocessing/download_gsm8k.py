#!/usr/bin/env python3
"""
Script to download the GSM8K dataset from Hugging Face and convert it to JSONL format.

Usage:
    python download_gsm8k.py --output_dir ./data --split train
    python download_gsm8k.py --output_dir ./data --split test
    python download_gsm8k.py --output_dir ./data --split all
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def download_and_convert_gsm8k(output_dir: str, split: str = "all"):
    """
    Download GSM8K dataset and convert to JSONL format.

    Args:
        output_dir: Directory to save the JSONL files
        split: Dataset split to download ('train', 'test', or 'all')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading GSM8K dataset from Hugging Face...")

    if split == "all":
        splits_to_process = ["train", "test"]
    else:
        splits_to_process = [split]

    for split_name in splits_to_process:
        print(f"\nProcessing {split_name} split...")

        # Load the dataset
        dataset = load_dataset("openai/gsm8k", "main", split=split_name)

        # Convert to JSONL
        output_file = output_path / f"gsm8k_{split_name}.jsonl"

        with open(output_file, "w", encoding="utf-8") as f:
            for example in dataset:
                answer = example["answer"].strip()
                del example["answer"]

                answer_idx = answer.find("####")
                if answer_idx == -1:
                    raise ValueError("Expected '####' in the answer field.")

                example["answer"] = answer[answer_idx + 4 :].strip()
                example["reasoning"] = answer[:answer_idx].strip()

                json_line = json.dumps(example, ensure_ascii=False)
                f.write(json_line + "\n")

        print(f"✓ Saved {len(dataset)} examples to {output_file}")

        # Print sample
        print(f"\nSample from {split_name} split:")
        print(json.dumps(dataset[0], indent=2, ensure_ascii=False))

    print(f"\n✓ All done! Files saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download GSM8K dataset and convert to JSONL format")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/gsm8k",
        help="Directory to save the JSONL files (default: ./data/gsm8k)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "all"],
        default="all",
        help="Dataset split to download (default: all)",
    )

    args = parser.parse_args()

    download_and_convert_gsm8k(args.output_dir, args.split)


if __name__ == "__main__":
    main()
