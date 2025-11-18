import ast
from typing import List, cast

import pandas as pd

answer_marker = ("[[", "]]")


def cot_sys_prompt(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += f" Explain your thinking process step-by-step. At the end, write down the number of the correct answer by strictly following this format: {answer_marker[0]}number_of_correct_answer{answer_marker[1]}."
    return sys_msg


def cot_sys_prompt_from_row(row: pd.Series):
    subject = row["base_cluster"]

    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += f" Explain your thinking process step-by-step. At the end, write down the number of the correct answer by strictly following this format: {answer_marker[0]}number_of_correct_answer{answer_marker[1]}."
    return sys_msg


def cot_sys_prompt_with_fallback_for_unknown_answers(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += f" Explain your thinking process step-by-step. At the end, if you are certain about the answer write down the number of the correct answer by strictly following this format: {answer_marker[0]}number_of_correct_answer{answer_marker[1]}, otherwise return {answer_marker[0]}0{answer_marker[1]}."
    return sys_msg


def cot_sys_prompt_with_fallback_for_unknown_answers_alternative(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += f" Explain your thinking process step-by-step. At the end, if you know the answer write down the number of the correct answer by strictly following this format: {answer_marker[0]}number_of_correct_answer{answer_marker[1]}, otherwise return {answer_marker[0]}0{answer_marker[1]}."
    return sys_msg


option_ids = [str(i + 1) for i in range(20)]


def cot_answer_prompt(question: str, options: List[str]):
    options_str = "\n".join([f"{option_id}. {answer}".strip() for option_id, answer in zip(option_ids, options)])
    user_prompt = f"Question: {question.strip()}\nOptions:\n{options_str}\n"
    return user_prompt


def cot_answer_prompt_from_row(row: pd.Series):
    question = cast(str, row["question"])
    options = ast.literal_eval(cast(str, row["options"]))

    options_str = "\n".join([f"{option_id}. {answer}".strip() for option_id, answer in zip(option_ids, options)])
    user_prompt = f"Question: {question.strip()}\nOptions:\n{options_str}\n"
    return user_prompt
