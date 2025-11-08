import sys
from pathlib import Path
from core.distillation.synth_aug_mmlu import synth_on_dataset

def _path_model(m: str) -> str:
    return (
        m.lower()
         .replace("/", "_")
         .replace(":", "_")
         .replace("-", "_")
         .replace(".", "_")
    )

def main():
    root = Path(__file__).resolve().parents[3]

    in_tsv = root / "data" / "source" / "mmlu_pro_stem.tsv"
    out_dir = root / "data" / "out" / "distillation"
    out_dir.mkdir(parents=True, exist_ok=True)

    models_to_branches = {
        "openai/gpt-oss-120b": ("A","B","C"),
        "qwen/qwen3-235b-a22b-thinking-2507": ("A", "B", "C"),
        "moonshotai/kimi-k2-thinking" : ("A", "B", "C")
    }

    limit = 100
    max_tokens = 16384
    dump_every = 20
    chunk_size = 30

    ds_stem = in_tsv.stem

    for model, branches in models_to_branches.items():
        model_path = _path_model(model)
        print(f"\n==> Model {model} | branches={branches}")

        for b in branches:
            out_name = f"{ds_stem}_synth_{model_path}_{b.lower()}_f{limit}.jsonl"
            out_jsonl = out_dir / out_name
            print(f"  -> Branch {b}: writing to {out_name}")

            synth_on_dataset(
                in_filename=str(in_tsv),
                out_jsonl=str(out_jsonl),
                model=model,
                max_tokens=max_tokens,
                dump_every=dump_every,
                limit=limit,
                branch=b,
                chunk_size=chunk_size,
            )

if __name__ == "__main__":
    main()
