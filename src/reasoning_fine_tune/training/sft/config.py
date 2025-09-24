from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TrainConfig:
    # paths
    train_path: str = "/complexity-aware-fine-tuning/data/data_splits/cross_entropy/phi/train_df_easy.tsv"
    valid_path: str = "/complexity-aware-fine-tuning/data/data_splits/cross_entropy/phi/valid_df_easy.tsv"
    test_path: str = "/complexity-aware-fine-tuning/data/data_splits/cross_entropy/phi/test_balanced_combined_entr.tsv"
    test_balanced_path: str = (
        "/complexity-aware-fine-tuning/data/data_splits/cross_entropy/phi/test_balanced_combined_entr.tsv"
    )
    save_dir: str = "/complexity-aware-fine-tuning/data/data_splits/cross_entropy/phi/"

    # model / training
    base_model: str = "/home/dviazhev/qa_finetune/Phi-4-mini-instruct"
    lr: float = 1e-5
    batch_size: int = 2
    gradient_accumulation: int = 8
    epochs: int = 3
    log_interval: int = 50
    debug: bool = False
    run_eval_on_start = True
    eval_validation_period = 1
    eval_test_period = 1
    eval_batch_size = 1

    # prompt / behaviour
    use_cot: bool = False
    max_input_length: int = 800
    max_new_tokens: int = 1500

    # runtime
    seed: int = 42
    num_workers: int = 4

    # model-specific prompt style (token wrappers)
    prompt_tokens: Dict[str, str] = field(
        default_factory=lambda: {
            "system_start": "<|system|>",
            "system_end": "<|endoftext|>",
            "user_start": "<|user|>",
            "user_end": "<|endofprompt|>",
            "assistant_start": "<|assistant|>",
            "assistant_end": "<|endoftext|>",
        }
    )

    @staticmethod
    def preset(name: str) -> "TrainConfig":
        # Define a couple of presets for different models (tweak tokens if necessary)
        if name.endswith("Phi-4-mini-instruct"):
            cfg = TrainConfig(base_model=name)
            cfg.prompt_tokens = {
                "system_start": "<|system|>",
                "system_end": "<|endoftext|>",
                "user_start": "<|user|>",
                "user_end": "<|endofprompt|>",
                "assistant_start": "<|assistant|>",
                "assistant_end": "<|endoftext|>",
            }
            return cfg
        if name.endswith("Qwen2.5-3B-Instruct"):
            cfg = TrainConfig(base_model=name)
            cfg.prompt_tokens = {
                "system_start": "<|im_start|>system",
                "system_end": "<|im_end|>",
                "user_start": "<|im_start|>user",
                "user_end": "<|im_end|>",
                "assistant_start": "<|im_start|>assistant",
                "assistant_end": "<|im_end|>",
            }
            return cfg
        # default
        return TrainConfig(base_model=name)
