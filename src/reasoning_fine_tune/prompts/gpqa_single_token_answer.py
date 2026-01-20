from typing import cast

import pandas as pd


def single_token_sys_prompt_from_row(row: pd.Series):
    sys_msg = "The following are multiple choice questions. Write down ONLY the LETTER of the correct answer and nothing else."
    return sys_msg


def single_token_answer_prompt_from_row(row: pd.Series):
    question = cast(str, row["question"])
    options = [f"{label}. {answer}" for label, answer in zip(row["option_ids"], row["options"])]

    options_str = "\n".join(options)
    user_prompt = f"Question: {question.strip()}\nOptions:\n{options_str}\nChoose one of the answers. Write down ONLY the LETTER of the correct answer and nothing else."
    return user_prompt
