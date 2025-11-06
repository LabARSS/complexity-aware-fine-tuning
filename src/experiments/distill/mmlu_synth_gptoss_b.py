from pathlib import Path
import os, sys

def main():
    root = Path(__file__).resolve().parents[3]

    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from core.distillation.synth_aug_branch_b import synth_on_dataset

    in_tsv = root / "data" / "source" / "mmlu_pro_stem.tsv"
    out_dir = root / "data" / "out" / "distillation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "mmlu_pro_synth_gptoss_f100b.jsonl"

    out_path = synth_on_dataset(
        in_filename=str(in_tsv),
        out_jsonl=str(out_jsonl),
        model="openai/gpt-oss-120b",
        max_tokens = 16384,
        dump_every = 20,
        limit = 100
    )
    print(out_path)

if __name__ == "__main__":
    main()
