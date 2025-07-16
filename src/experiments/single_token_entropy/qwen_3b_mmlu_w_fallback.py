import ast
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning_fine_tune.entropy_estimation.estimate_single_token_entropy import estimate_dataset
from reasoning_fine_tune.prompts.mmlu_single_token_answer import (
    single_token_sys_prompt_with_fallback_for_unknown_answers,
)
from reasoning_fine_tune.utils.correctness import check_answer_correct_mmlu
from reasoning_fine_tune.utils.device import DEVICE_MAP

print(f"Using device: {DEVICE_MAP}")

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=DEVICE_MAP, torch_dtype=torch.bfloat16)

inferred_device_map = model.hf_device_map
print("\nInferred Device Map:", inferred_device_map)


estimate_dataset(
    in_filename=Path(__file__).parent.joinpath("../../../data/source/mmlu_pro_stem.tsv").resolve(),
    out_filename=Path(__file__)
    .parent.joinpath("../../../data/out/single_token_entropy/mmlu_qwen_3b_w_fallback_if_unknown.tsv")
    .resolve(),
    model=model,
    tokenizer=tokenizer,
    get_subject_from_row=lambda row: row["base_cluster"],
    get_question_from_row=lambda row: row["question"],
    get_options_from_row=lambda row: ast.literal_eval(row["options"]),
    check_answer_correct=check_answer_correct_mmlu,
    get_sys_prompt=single_token_sys_prompt_with_fallback_for_unknown_answers,
    batch_size=256
)
