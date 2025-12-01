import ast
from pathlib import Path

from core.complexity_estimation.entropy.distill_entropy import distill_logprobs
from core.utils.correctness import check_answer_correct_mmlu

distill_logprobs(
    in_filename=Path(__file__).parent.joinpath("../../../data/source/mmlu_pro_stem.tsv").resolve(),
    out_filename=Path(__file__)
    .parent.joinpath("../../../data/out/single_token_entropy/mmlu_large_llm_combined.parquet")
    .resolve(),
    model="deepseek/deepseek-chat-v3.1",
    provider="gmicloud/fp8",
    get_subject_from_row=lambda row: row["base_cluster"],
    get_question_from_row=lambda row: row["question"],
    get_options_from_row=lambda row: ast.literal_eval(row["options"]),
    check_answer_correct=check_answer_correct_mmlu,
)
