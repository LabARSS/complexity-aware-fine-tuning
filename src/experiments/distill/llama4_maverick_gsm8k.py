from multiprocessing import freeze_support
from pathlib import Path
import pandas as pd
from typing import Callable

from reasoning_fine_tune.distillation.distill import DistillConfig, distill_on_dataset
from reasoning_fine_tune.prompts.gsm8k_cot_answer import cot_answer_prompt_from_row, cot_sys_prompt_from_row
from reasoning_fine_tune.utils.correctness import check_answer_correct_gsm8k


class MaverickConfig(DistillConfig):
    in_filename: str = str(Path(__file__).parent.joinpath("../../../data/source/gsm8k/gsm8k_train.jsonl").resolve())
    out_filename: str = str(Path(__file__).parent.joinpath("../../../data/out/distillation/gsm8k_maverick.jsonl").resolve())
    model: str = "meta-llama/llama-4-maverick"
    check_answer_correct: Callable[[pd.Series, str], bool] = check_answer_correct_gsm8k
    get_sys_prompt: Callable[[pd.Series], str] = cot_sys_prompt_from_row
    get_user_prompt: Callable[[pd.Series], str] = cot_answer_prompt_from_row
    dump_every: int = 100


if __name__ == "__main__":
    freeze_support()

    distill_on_dataset(config=MaverickConfig())
