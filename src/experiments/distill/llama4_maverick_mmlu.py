import ast
from multiprocessing import freeze_support
from pathlib import Path

from reasoning_fine_tune.distillation.distill import distill_on_dataset
from reasoning_fine_tune.utils.correctness import check_answer_correct_mmlu

if __name__ == "__main__":
    freeze_support()

    distill_on_dataset(
        in_filename=Path(__file__).parent.joinpath("../../../data/source/mmlu_pro_stem_shuffled.tsv").resolve(),
        out_filename=Path(__file__).parent.joinpath("../../../data/out/distillation/mmlu_merged.tsv").resolve(),
        get_subject_from_row=lambda row: row["base_cluster"],
        get_question_from_row=lambda row: row["question"],
        get_options_from_row=lambda row: ast.literal_eval(row["options"]),
        check_answer_correct=check_answer_correct_mmlu,
        model="meta-llama/llama-4-maverick",
    )
