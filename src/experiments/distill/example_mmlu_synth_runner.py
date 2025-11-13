from pathlib import Path
import sys

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
    out_dir = root / "data" / "out"  / "distillation"
    out_dir.mkdir(parents=True, exist_ok=True)

    models_to_branches = {
        "openai/gpt-oss-120b": ("A","B"),
        "qwen/qwen3-235b-a22b-thinking-2507": ("A", "B"),
        "moonshotai/kimi-k2-thinking" : ("A", "B")
    }

    limit = 10
    max_tokens = 16384
    dump_every = 2
    chunk_size = 10
    temps = {"A": 0.0, "B": 0.0, "C": 0.0}

    ds_stem = in_tsv.stem

    for model, branches in models_to_branches.items():
        model_path = _path_model(model)
        print(f"\n==> Model {model} | branches={branches}")

        #FIRSTLY A, then C
        a_file = out_dir / f"{ds_stem}_synth_{model_path}_a_f{limit}.jsonl"

        for b in branches:
            out_name = f"{ds_stem}_synth_{model_path}_{b.lower()}_f{limit}.jsonl"
            out_jsonl = out_dir / out_name
            print(f"  -> Branch {b}: writing to {out_name}")

            a_jsonl_path = str(a_file) if b == "C" else None

            synth_on_dataset(
                in_filename=str(in_tsv),
                out_jsonl=str(out_jsonl),
                model=model,
                max_tokens=max_tokens,
                dump_every=dump_every,
                limit=limit,
                branch=b,
                chunk_size=chunk_size,
                a_jsonl_path=a_jsonl_path,
                temperature=temps[b],
            )

if __name__ == "__main__":
    main()
