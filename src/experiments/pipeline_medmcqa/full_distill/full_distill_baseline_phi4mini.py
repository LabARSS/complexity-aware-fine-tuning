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
    cfg.max_input_length = 6000
    cfg.max_input_length_eval = 1024
    cfg.max_new_tokens = 5000
    cfg.batch_size = 4
    cfg.eval_batch_size = 16
    cfg.gradient_accumulation = 64
    cfg.train_sample_size = None
    cfg.val_sample_size = None
    cfg.test_sample_size = None
    cfg.dataset = "medmcqa"

    cfg.save_dir = str(
        Path(__file__).parent.joinpath("../../../../artifacts/pipeline_medmcqa/full_distill_baseline/phi4")
    )
    cfg.train_path = str(
        Path(__file__).parent.joinpath("../../../../data/out/distillation/medmcqa_deepseek_v3_filtered.jsonl")
    )
    cfg.valid_path = str(Path(__file__).parent.joinpath("../../../../data/source/medmcqa/medmcqa_test.jsonl"))
    cfg.test_path = str(Path(__file__).parent.joinpath("../../../../data/source/medmcqa/medmcqa_test.jsonl"))
    cfg.test_balanced_path = str(Path(__file__).parent.joinpath("../../../../data/source/medmcqa/medmcqa_test.jsonl"))

    trainer = Trainer(cfg)
    trainer.train()
