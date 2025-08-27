from multiprocessing import freeze_support
from pathlib import Path

from core.training.sft_by_complexity_split.sft_by_all_complexity_splits import (
    train_sft_by_all_complexity_splits,
)

if __name__ == "__main__":
    freeze_support()

    train_sft_by_all_complexity_splits(
        data_folder_path=str(
            Path(__file__).parent.joinpath("../../../data/out/splits/cross_entropy/qwen_3b/").resolve()
        ),
        out_path=str(Path(__file__).parent.joinpath("../../../artifacts/cross_entropy/qwen3b/").resolve()),
        model_id="Qwen/Qwen2.5-3B-Instruct",
    )
