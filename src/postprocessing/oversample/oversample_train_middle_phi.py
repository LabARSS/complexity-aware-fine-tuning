import ast
from pathlib import Path

from reasoning_fine_tune.processing.oversample import oversample_dataset

oversample_dataset(
    in_filename=Path(__file__).parent.joinpath("../../../data/data_splits/entropy_fallback/phi/train_df_hard.tsv").resolve(),
    out_filename=Path(__file__)
    .parent.joinpath("../../../data/data_splits/entropy_fallback/phi/train_df_hard_oversampled.tsv")
    .resolve(),
    question_col="question",
    get_options_from_row=lambda row: ast.literal_eval(row["options"]),
    oversample_ratio=2,
)
