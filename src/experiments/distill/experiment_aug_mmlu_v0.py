from pathlib import Path
import os, sys

def main():
    root = Path(__file__).resolve().parents[3]

    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from core.distillation.synth_aug_mmlu import synth_on_dataset

    in_tsv = root / "data" / "source" / "mmlu_pro_stem.tsv"
    out_dir = root / "data" / "out" / "distillation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "mmlu_pro_stem.v0.jsonl"

    model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-0528")
    out_path = synth_on_dataset(
        in_filename=str(in_tsv),
        out_jsonl=str(out_jsonl),
        model=model,
        max_tokens=int(os.getenv("OPENROUTER_MAX_TOKENS", "1024")),
        dump_every=2,
        limit=10,
        branches=("B", "C"),
    )
    print(out_path)

if __name__ == "__main__":
    main()
