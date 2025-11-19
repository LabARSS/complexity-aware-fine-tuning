from pathlib import Path
from typing import Callable

import pandas as pd
import torch

from reasoning_fine_tune.entropy_estimation.estimate_single_token_entropy import EstimateDatasetConfig, estimate_dataset
from reasoning_fine_tune.prompts.gsm8k_cot_answer import cot_answer_prompt_from_row, cot_sys_prompt_from_row
from reasoning_fine_tune.utils.correctness import check_answer_correct_gsm8k


class Phi4Config(EstimateDatasetConfig):
    in_filename: str = str(Path(__file__).parent.joinpath("../../../data/source/gsm8k/gsm8k_train.jsonl").resolve())
    out_filename: str = str(
        Path(__file__).parent.joinpath("../../../data/out/single_token_entropy/gsm8k_phi4mini.jsonl").resolve()
    )
    dump_every: int = 100
    max_new_tokens: int = 16384
    get_sys_prompt: Callable[[pd.Series], str] = cot_sys_prompt_from_row
    get_user_prompt: Callable[[pd.Series], str] = cot_answer_prompt_from_row
    model_name: str = "microsoft/phi-4"
    device: str = "cuda"
    check_answer_correct: Callable[[pd.Series, str], bool] = check_answer_correct_gsm8k
    model_config_dict: dict = {"torch_dtype": torch.bfloat16}


estimate_dataset(config=Phi4Config())
