from pathlib import Path

from reasoning_fine_tune.training.sft.config import TrainConfig
from reasoning_fine_tune.training.sft.train import Trainer

if __name__ == "__main__":
    cfg = TrainConfig.preset("microsoft/Phi-4-mini-instruct")
    cfg.epochs = 20
    cfg.debug = False
    cfg.use_cot = True
    cfg.eval_validation_period = 4
    cfg.eval_test_period = 0
    cfg.run_eval_on_start = False
    cfg.max_input_length = 8192
    cfg.max_input_length_eval = 1024
    cfg.max_new_tokens = 8192
    cfg.batch_size = 4
    cfg.eval_batch_size = 16
    cfg.gradient_accumulation = 64
    cfg.train_sample_size = None
    cfg.val_sample_size = None
    cfg.test_sample_size = None
    cfg.dataset = "gsm8k"

    cfg.save_dir = str(
        Path(__file__).parent.joinpath("../../../../artifacts/pipeline_gsm8k/full_distill_baseline/phi4")
    )
    cfg.train_path = str(Path(__file__).parent.joinpath("../../../../data/out/distillation/gsm8k_distilled_cot.jsonl"))
    cfg.valid_path = str(Path(__file__).parent.joinpath("../../../../data/source/gsm8k/gsm8k_test.jsonl"))
    cfg.test_path = str(Path(__file__).parent.joinpath("../../../../data/source/gsm8k/gsm8k_test.jsonl"))
    cfg.test_balanced_path = str(Path(__file__).parent.joinpath("../../../../data/source/gsm8k/gsm8k_test.jsonl"))

    trainer = Trainer(cfg)
    trainer.train()
