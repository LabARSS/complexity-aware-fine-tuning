from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TrainConfig:
    # paths
    train_path: str | list[str] = "/complexity-aware-fine-tuning/data/data_splits/cross_entropy/phi/train_df_easy.tsv"
    valid_path: str = "/complexity-aware-fine-tuning/data/data_splits/cross_entropy/phi/valid_df_easy.tsv"
    test_path: str = "/complexity-aware-fine-tuning/data/data_splits/cross_entropy/phi/test_balanced_combined_entr.tsv"
    test_balanced_path: str = (
        "/complexity-aware-fine-tuning/data/data_splits/cross_entropy/phi/test_balanced_combined_entr.tsv"
    )
    save_dir: str = "/complexity-aware-fine-tuning/data/data_splits/cross_entropy/phi/"

    # model / training
    # model / training
    base_model: str = "/home/dviazhev/qa_finetune/Phi-4-mini-instruct"
    lr: float = 1e-4
    batch_size: int = 8
    gradient_accumulation: int = 32
    epochs: int = 3
    log_interval: int = 50
    debug: bool = True
    run_eval_on_start = True
    eval_validation_period = 1
    eval_test_period: int | list[int] = 25
    eval_batch_size = 1

    train_sample_size: int | None = 3000
    val_sample_size: int | None = 1000
    test_sample_size: int | None = 1000

    # prompt / behaviour
    use_cot: bool = False
    max_input_length: int = 2048
    max_input_length_eval: int = 2048
    max_new_tokens: int = 30

    # runtime
    seed: int = 42
    num_workers: int = 4

    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    dataset: str = "mmlu"  # medmcqa, mmlu, gsmk

    lora_target_modules: tuple[str, ...] = field(
        default_factory=lambda: (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
    )

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
        if name.endswith("Llama-3.2-3B-Instruct"):
            cfg = TrainConfig(base_model=name)
            cfg.prompt_tokens = {
                "system_start": "<|start_header_id|>system<|end_header_id|>\n\n",
                "system_end": "<|eot_id|>",
                "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
                "user_end": "<|eot_id|>",
                "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
                "assistant_end": "<|eot_id|>",
            }
            return cfg
        # default
        return TrainConfig(base_model=name)
