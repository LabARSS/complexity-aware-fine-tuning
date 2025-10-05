from pathlib import Path

from reasoning_fine_tune.training.sft.config import TrainConfig
from reasoning_fine_tune.training.sft.train import Trainer

if __name__ == "__main__":
    cfg = TrainConfig.preset("microsoft/Phi-4-mini-instruct")
    cfg.epochs = 10
    cfg.debug = False

    cfg.save_dir = str(Path(__file__).parent.joinpath("../../../../artifacts/pipeline_20epochs/alternative/phi4/pt1"))
    cfg.train_path = [
        str(Path(__file__).parent.joinpath("../../../../data/data_splits/entropy_fallback/phi/train_df_hard.tsv")),
    ]
    cfg.valid_path = str(
        Path(__file__).parent.joinpath("../../../../data/data_splits/entropy_fallback/phi/valid_df_combined.tsv")
    )
    cfg.test_path = str(
        Path(__file__).parent.joinpath(
            "../../../../data/data_splits/entropy_fallback/phi/test_balanced_combined_entr.tsv"
        )
    )
    cfg.test_balanced_path = str(
        Path(__file__).parent.joinpath(
            "../../../../data/data_splits/entropy_fallback/phi/test_balanced_combined_entr.tsv"
        )
    )

    trainer = Trainer(cfg)
    trainer.train(save=True)
