from pathlib import Path

from reasoning_fine_tune.training.sft.config import TrainConfig
from reasoning_fine_tune.training.sft.train import Trainer

if __name__ == "__main__":
    cfg = TrainConfig.preset("microsoft/Phi-4-mini-instruct")
    cfg.epochs = 20
    cfg.debug = False

    cfg.save_dir = str(Path(__file__).parent.joinpath("../../../../artifacts/pipeline_gsm8k/sft_baseline/phi4"))
    cfg.train_path = str(Path(__file__).parent.joinpath("../../../../data/source/gsm8k/gsm8k_train.jsonl"))
    cfg.valid_path = str(Path(__file__).parent.joinpath("../../../../data/source/gsm8k/gsm8k_test.jsonl"))
    cfg.test_path = str(Path(__file__).parent.joinpath("../../../../data/source/gsm8k/gsm8k_test.jsonl"))
    cfg.test_balanced_path = str(Path(__file__).parent.joinpath("../../../../data/source/gsm8k/gsm8k_test.jsonl"))

    trainer = Trainer(cfg)
    trainer.train()
