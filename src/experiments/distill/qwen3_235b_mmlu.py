from multiprocessing import freeze_support
from pathlib import Path

from reasoning_fine_tune.distillation.distill import DistillConfig, distill_on_dataset
from reasoning_fine_tune.prompts.mmlu_cot_answer import cot_answer_prompt_from_row, cot_sys_prompt_from_row
from reasoning_fine_tune.utils.correctness import check_answer_correct_mmlu


class Qwen3Config(DistillConfig):
    in_filename = str(Path(__file__).parent.joinpath("../../../data/source/mmlu_pro_stem_shuffled.tsv").resolve())
    out_filename = str(Path(__file__).parent.joinpath("../../../data/out/distillation/mmlu_qwen3_235b.tsv").resolve())
    model = "qwen/qwen3-235b-a22b-2507"
    check_answer_correct = check_answer_correct_mmlu
    get_sys_prompt = cot_sys_prompt_from_row
    get_user_prompt = cot_answer_prompt_from_row


if __name__ == "__main__":
    freeze_support()

    distill_on_dataset(config=Qwen3Config())
