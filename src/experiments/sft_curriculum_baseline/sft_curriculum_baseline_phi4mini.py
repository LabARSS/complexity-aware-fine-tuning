from pathlib import Path

import numpy as np
import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from reasoning_fine_tune.training.sft_curriculum import train_sft_curriculum
from reasoning_fine_tune.utils.device import DEVICE_MAP

np.random.seed(42)
torch.manual_seed(42)


print(f"Using device: {DEVICE_MAP}")

MODEL_NAME = "microsoft/Phi-4-mini-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=DEVICE_MAP)

inferred_device_map = model.hf_device_map
print("\nInferred Device Map:", inferred_device_map)

train_sft_curriculum(
    name="qwen3b",
    model=model,
    tokenizer=tokenizer,
    easy_train_df_path=Path(__file__)
    .parent.joinpath("../../../data/data_splits/entropy_fallback/phi/train_df_easy.tsv")
    .resolve(),
    mid_train_df_path=Path(__file__)
    .parent.joinpath("../../../data/data_splits/entropy_fallback/phi/train_df_middle.tsv")
    .resolve(),
    hard_train_df_path=Path(__file__)
    .parent.joinpath("../../../data/data_splits/entropy_fallback/phi/train_df_hard.tsv")
    .resolve(),
    test_df_path=Path(__file__)
    .parent.joinpath("../../../data/data_splits/entropy_fallback/phi/test_balanced_combined_entr.tsv")
    .resolve(),
)
