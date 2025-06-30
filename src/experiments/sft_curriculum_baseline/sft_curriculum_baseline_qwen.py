from pathlib import Path

from reasoning_fine_tune.training.sft_curriculum import train_sft_curriculum

if __name__ == "__main__": 
    train_sft_curriculum(
        name="qwen3b",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        easy_train_df_path=Path(__file__)
        .parent.joinpath("../../../data/data_splits/entropy_fallback/qwen/train_df_easy.tsv")
        .resolve(),
        mid_train_df_path=Path(__file__)
        .parent.joinpath("../../../data/data_splits/entropy_fallback/qwen/train_df_middle.tsv")
        .resolve(),
        hard_train_df_path=Path(__file__)
        .parent.joinpath("../../../data/data_splits/entropy_fallback/qwen/train_df_hard.tsv")
        .resolve(),
        test_df_path=Path(__file__)
        .parent.joinpath("../../../data/data_splits/entropy_fallback/qwen/test_balanced_combined_entr.tsv")
        .resolve(),
    )
