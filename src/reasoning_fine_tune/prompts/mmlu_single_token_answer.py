import ast
import pandas as pd
from typing import List, cast


def single_token_sys_prompt(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += " Write down ONLY the NUMBER of the correct answer and nothing else."
    return sys_msg


def single_token_sys_prompt_from_row(row: pd.Series):
    subject = row["base_cluster"]

    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += " Write down ONLY the NUMBER of the correct answer and nothing else."
    return sys_msg


def single_token_sys_prompt_with_fallback_for_unknown_answers(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += " If you are certain about the answer return the correct option number, otherwise return 0. Write down ONLY the NUMBER and nothing else."
    return sys_msg


def single_token_sys_prompt_with_fallback_for_unknown_answers_alternative(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += " If you know the answer return the correct option number, otherwise return 0. Write down ONLY the NUMBER and nothing else."
    return sys_msg


option_ids = [str(i + 1) for i in range(20)]
# 0 is a special exception for "do not know"
option_ids_w_fallback = option_ids + ["0"]


def single_token_answer_prompt(question: str, options: List[str]):
    options_str = "\n".join([f"{option_id}. {answer}".strip() for option_id, answer in zip(option_ids, options)])
    user_prompt = f"Question: {question.strip()}\nOptions:\n{options_str}\nChoose one of the answers. Write down ONLY the NUMBER of the correct answer and nothing else."
    return user_prompt


def single_token_answer_prompt_from_row(row: pd.Series):
    question = cast(str, row["question"])
    options = ast.literal_eval(cast(str, row["options"]))

    options_str = "\n".join([f"{option_id}. {answer}".strip() for option_id, answer in zip(option_ids, options)])
    user_prompt = f"Question: {question.strip()}\nOptions:\n{options_str}\nChoose one of the answers. Write down ONLY the NUMBER of the correct answer and nothing else."
    return user_prompt


def single_token_answer_prompt_with_fallback_for_unknown_answers(question: str, options: List[str]):
    options_str = "\n".join([f"{option_id}. {answer}".strip() for option_id, answer in zip(option_ids, options)])
    user_prompt = f"Question: {question.strip()}\nOptions:\n{options_str}\nChoose one of the answers. If you are certain about the answer return the correct option number, otherwise return 0. Write down ONLY the NUMBER and nothing else."
    return user_prompt


def single_token_answer_prompt_with_fallback_for_unknown_answers_alternative(question: str, options: List[str]):
    options_str = "\n".join([f"{option_id}. {answer}".strip() for option_id, answer in zip(option_ids, options)])
    user_prompt = f"Question: {question.strip()}\nOptions:\n{options_str}\nChoose one of the answers. If you know the answer return the correct option number, otherwise return 0. Write down ONLY the NUMBER and nothing else."
    return user_prompt
