#!/usr/bin/env python3
"""Count accuracy for JSONL files with gold and answer fields."""

import argparse
import json
from pathlib import Path


def count_accuracy(jsonl_path: str) -> None:
    """Count accuracy for entries in a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file
    """
    path = Path(jsonl_path)

    if not path.exists():
        print(f"Error: File not found: {jsonl_path}")
        return

    total = 0
    correct = 0
    malformed = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                total += 1

                gold = entry["input"]["gold"]
                answer = entry["output"]["answer"]

                if gold == answer:
                    correct += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                malformed += 1
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                malformed += 1
                continue

    if total == 0:
        print("No valid entries found in the file.")
        if malformed > 0:
            print(f"Malformed entries: {malformed}")
        return

    accuracy = (correct / total) * 100

    print(f"Total entries: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    if malformed > 0:
        print(f"Malformed entries: {malformed}")


def main():
    parser = argparse.ArgumentParser(description="Count accuracy for JSONL files with gold and answer fields")
    parser.add_argument("jsonl_path", type=str, help="Path to the JSONL file")

    args = parser.parse_args()
    count_accuracy(args.jsonl_path)


if __name__ == "__main__":
    main()
