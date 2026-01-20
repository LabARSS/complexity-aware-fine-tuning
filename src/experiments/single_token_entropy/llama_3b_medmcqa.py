from pathlib import Path

import pandas as pd
import torch

from reasoning_fine_tune.entropy_estimation.estimate_single_token_entropy import (
    Callable,
    EstimateDatasetConfig,
    estimate_dataset,
)
from reasoning_fine_tune.prompts.medmcqa_single_token_answer import (
    single_token_answer_prompt_from_row,
    single_token_sys_prompt_from_row,
)
from reasoning_fine_tune.utils.correctness import check_answer_correct_medmcqa


class LLama3BConfig(EstimateDatasetConfig):
    in_filename: str = str(Path(__file__).parent.joinpath("../../../data/source/medmcqa/medmcqa_train.jsonl").resolve())
    out_filename: str = str(
        Path(__file__).parent.joinpath("../../../data/out/single_token_entropy/medmcqa_llama_3b.jsonl").resolve()
    )
    dump_every: int = 100
    max_new_tokens: int = 1
    get_sys_prompt: Callable[[pd.Series], str] = single_token_sys_prompt_from_row
    get_user_prompt: Callable[[pd.Series], str] = single_token_answer_prompt_from_row
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    device: str = "cuda"
    check_answer_correct: Callable[[pd.Series, str], bool] = check_answer_correct_medmcqa
    model_config_dict: dict = {"torch_dtype": torch.bfloat16}


estimate_dataset(LLama3BConfig())
