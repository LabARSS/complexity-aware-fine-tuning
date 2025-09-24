from pathlib import Path

from reasoning_fine_tune.training.sft.config import TrainConfig
from reasoning_fine_tune.training.sft.train import Trainer

if __name__ == "__main__":
    cfg = TrainConfig.preset("Qwen/Qwen2.5-3B-Instruct")
    cfg.epochs = 20
    cfg.debug = False
    cfg.use_cot = True
    cfg.eval_validation_period = 0
    cfg.eval_test_period = 5
    cfg.run_eval_on_start = False
    cfg.batch_size = 4
    cfg.max_new_tokens = 8192
    cfg.eval_batch_size = 8

    cfg.save_dir = str(
        Path(__file__).parent.joinpath("../../../../artifacts/pipeline_20epochs/full_distill_baseline/qwen")
    )
    cfg.train_path = str(
        Path(__file__).parent.joinpath("../../../../data/data_splits/entropy_fallback/qwen/train_df_combined.tsv")
    )
    cfg.valid_path = str(
        Path(__file__).parent.joinpath("../../../../data/data_splits/entropy_fallback/qwen/valid_df_combined.tsv")
    )
    cfg.test_path = str(
        Path(__file__).parent.joinpath(
            "../../../../data/data_splits/entropy_fallback/qwen/test_balanced_combined_entr.tsv"
        )
    )
    cfg.test_balanced_path = str(
        Path(__file__).parent.joinpath(
            "../../../../data/data_splits/entropy_fallback/qwen/test_balanced_combined_entr.tsv"
        )
    )

    trainer = Trainer(cfg)
    trainer.train()
