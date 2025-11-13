from pathlib import Path
from multiprocessing import freeze_support

from core.distillation.synth_aug_mmlu import synth_on_dataset

def main():
    freeze_support()

    synth_on_dataset(
        in_filename=Path(__file__).parent.joinpath("../../../data/source/mmlu_pro_stem_shuffled.tsv").resolve(),
        out_jsonl=Path(__file__).parent.joinpath("../../../data/out/distillation/temperature_changes/mmlu_synth_gptoss_a_t0_8.jsonl").resolve(),
        model="openai/gpt-oss-120b",
        max_tokens=16384,
        dump_every=1000,
        limit=None,
        branch="A",
        chunk_size=30,
        a_jsonl_path=None, # for branch C
        temperature=0.8,
    )

if __name__ == "__main__":
    main()