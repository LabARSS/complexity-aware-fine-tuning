from multiprocessing import freeze_support
from pathlib import Path
import pandas as pd

from reasoning_fine_tune.distillation.distill import Callable, DistillConfig, distill_on_dataset
from reasoning_fine_tune.prompts.gsm8k_cot_answer import cot_answer_prompt_from_row, cot_sys_prompt_from_row
from reasoning_fine_tune.utils.correctness import check_answer_correct_gsm8k


class V3Config(DistillConfig):
    in_filename: str = str(Path(__file__).parent.joinpath("../../../data/out/distillation/gsm8k_all.jsonl").resolve())
    out_filename: str = str(Path(__file__).parent.joinpath("../../../data/out/distillation/gsm8k_all.jsonl").resolve())
    model: str = "moonshotai/kimi-k2"
    check_answer_correct: Callable[[pd.Series, str], bool] = check_answer_correct_gsm8k
    get_sys_prompt: Callable[[pd.Series], str] = cot_sys_prompt_from_row
    get_user_prompt: Callable[[pd.Series], str] = cot_answer_prompt_from_row


if __name__ == "__main__":
    freeze_support()

    distill_on_dataset(config=V3Config())
