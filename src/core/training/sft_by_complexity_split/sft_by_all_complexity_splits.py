from pathlib import Path

from core.training.sft_by_complexity_split.sft_by_single_complexity_split import (
    train_sft_by_complexity_split,
)
from core.utils.training_memory_sandbox import memory_sandbox_worker, run_in_memory_sandbox


TRAIN_POSTFIX = "_train.tsv"
TEST_POSTFIX = "_test.tsv"


@memory_sandbox_worker
def wrapped_train_sft_by_complexity_split(*args, **kwargs):
    return train_sft_by_complexity_split(*args, **kwargs)


def train_sft_by_all_complexity_splits(
    data_folder_path: str,
    out_path: str,
    model_id: str,
    training_kwargs: dict | None = None,
    *,                         
    use_lora: bool = False,      
    lora_kwargs: dict | None = None, 
):
    p = Path(data_folder_path)

    if not p.is_dir():
        raise NotADirectoryError(f"{p} is not a directory")

    children = [c for c in p.iterdir()]
    children.sort()  # alphabetical, case-sensitive

    train_df_paths = [c for c in children if c.name.endswith(TRAIN_POSTFIX)]
    test_df_paths = [c for c in children if c.name.endswith(TEST_POSTFIX)]

    for i, train_df_path in enumerate(train_df_paths):
        split_out_path = Path(out_path).joinpath(f"g{i}")
        print("train_sft_by_complexity_split", train_df_path)
        run_in_memory_sandbox(
            wrapped_train_sft_by_complexity_split,
            out_path=split_out_path,
            model_id=model_id,
            train_df_path=train_df_path,
            test_df_paths=test_df_paths,
            training_kwargs=training_kwargs,
            use_lora=use_lora,           
            lora_kwargs=lora_kwargs,     
        )