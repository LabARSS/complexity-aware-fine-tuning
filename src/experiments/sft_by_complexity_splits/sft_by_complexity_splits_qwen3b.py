from multiprocessing import freeze_support
from pathlib import Path

from core.training.sft_by_complexity_split.sft_by_all_complexity_splits import (
    train_sft_by_all_complexity_splits,
)

if __name__ == "__main__":
    freeze_support()

    train_sft_by_all_complexity_splits(
        data_folder_path=str(
            Path(__file__).parent.joinpath("../../../data/out/splits/single_token_entropy/qwen_3b/").resolve()
        ),
        out_path=str(
            Path(__file__).parent.joinpath("../../../artifacts/sft_by_complexity_split_lora/qwen3b/").resolve()
        ),
        model_id="Qwen/Qwen2.5-3B-Instruct",
        use_lora=True,
        training_kwargs={
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
        },
    )
