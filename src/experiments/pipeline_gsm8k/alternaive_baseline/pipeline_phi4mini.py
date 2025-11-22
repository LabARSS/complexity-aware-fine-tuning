from pathlib import Path

from reasoning_fine_tune.training.sft.config import TrainConfig
from reasoning_fine_tune.training.sft.train import Trainer

if __name__ == "__main__":
    cfg = TrainConfig.preset("microsoft/Phi-4-mini-instruct")
    cfg.epochs = 20
    cfg.debug = False
    cfg.use_cot = True
    cfg.eval_validation_period = 1
    cfg.eval_test_period = 0
    cfg.run_eval_on_start = False
    cfg.max_input_length = 16384
    cfg.max_new_tokens = 16384
    cfg.batch_size = 4
    cfg.gradient_accumulation = 64

    cfg.save_dir = str(Path(__file__).parent.joinpath("../../../../artifacts/pipeline_gsm8k/pipeline/phi4"))
    cfg.train_path = str(Path(__file__).parent.joinpath("../../../../data/data_splits/entropy/phi/gsm8k_hard.jsonl"))
    cfg.valid_path = str(Path(__file__).parent.joinpath("../../../../data/source/gsm8k/gsm8k_test.jsonl"))
    cfg.test_path = str(Path(__file__).parent.joinpath("../../../../data/source/gsm8k/gsm8k_test.jsonl"))
    cfg.test_balanced_path = str(Path(__file__).parent.joinpath("../../../../data/source/gsm8k/gsm8k_test.jsonl"))

    trainer = Trainer(cfg)
    trainer.train()
