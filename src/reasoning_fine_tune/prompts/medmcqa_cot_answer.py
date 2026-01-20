from typing import cast

import pandas as pd

answer_marker = ("[[", "]]")


def cot_sys_prompt_from_row(_row: pd.Series):
    sys_msg = f"The following are multiple choice questions. Explain your thinking process step-by-step. At the end, write down the LETTER of the correct answer by strictly following this format: {answer_marker[0]}letter_of_correct_answer{answer_marker[1]}."
    return sys_msg


def cot_answer_prompt_from_row(row: pd.Series):
    question = cast(str, row["question"])
    options = [f"{label}. {answer}" for label, answer in zip(row["option_ids"], row["options"])]

    options_str = "\n".join(options)
    user_prompt = f"Question: {question.strip()}\nOptions:\n{options_str}\n"
    return user_prompt
