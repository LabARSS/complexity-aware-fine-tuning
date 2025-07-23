from pathlib import Path

from reasoning_fine_tune.training.sft_by_complexity_split.sft_by_single_complexity_split import (
    train_sft_by_complexity_split,
)

TRAIN_POSTFIX = "_train.tsv"
TEST_POSTFIX = "_test.tsv"


def group_train_sft_by_complexity_split(data_folder_path: str, out_path: str, model_id: str, training_kwargs):
    p = Path(data_folder_path)

    if not p.is_dir():
        raise NotADirectoryError(f"{p} is not a directory")

    children = [c for c in p.iterdir()]
    children.sort()  # alphabetical, case-sensitive

    train_df_paths = [c for c in children if c.name.endswith(TRAIN_POSTFIX)]
    test_df_paths = [c for c in children if c.name.endswith(TEST_POSTFIX)]

    for i, train_df_path in enumerate(train_df_paths):
        split_out_path = Path(out_path).joinpath(f"g{i}")
        train_sft_by_complexity_split(
            out_path=split_out_path,
            model_id=model_id,
            train_df_path=train_df_path,
            test_df_paths=test_df_paths,
            training_kwargs=training_kwargs,
        )
