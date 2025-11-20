from pathlib import Path

from reasoning_fine_tune.training.sft.config import TrainConfig
from reasoning_fine_tune.training.sft.train import Trainer

if __name__ == "__main__":
    cfg = TrainConfig.preset("Qwen/Qwen2.5-3B-Instruct")
    cfg.epochs = 20
    cfg.debug = False
    cfg.use_cot = True
    cfg.eval_validation_period = 1
    cfg.eval_test_period = 0
    cfg.run_eval_on_start = False

    cfg.save_dir = str(
        Path(__file__).parent.joinpath("../../../../artifacts/pipeline_gsm8k/full_distill_baseline/qwen3b")
    )
    cfg.train_path = str(Path(__file__).parent.joinpath("../../../../data/out/distillation/gsm8k_distilled_cot.jsonl"))
    cfg.valid_path = str(Path(__file__).parent.joinpath("../../../../data/source/gsm8k/gsm8k_test.jsonl"))
    cfg.test_path = str(Path(__file__).parent.joinpath("../../../../data/source/gsm8k/gsm8k_test.jsonl"))
    cfg.test_balanced_path = str(Path(__file__).parent.joinpath("../../../../data/source/gsm8k/gsm8k_test.jsonl"))

    trainer = Trainer(cfg)
    trainer.train()
