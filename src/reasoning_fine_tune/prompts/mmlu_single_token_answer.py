from typing import List

from reasoning_fine_tune.prompts.mmlu_option_ids import fallback_option_id, option_ids


def single_token_sys_prompt(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += " Choose a correct option letter. Answer with a single symbol. Do not print anything else."
    return sys_msg


def single_token_sys_prompt_with_fallback_for_unknown_answers(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += f" If you are certain about the answer choose a correct option letter, otherwise return {fallback_option_id}. Answer with a single symbol. Do not print anything else."
    return sys_msg


def single_token_sys_prompt_with_fallback_for_unknown_answers_alternative(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += f" If you know the answer choose a correct option letter, otherwise return {fallback_option_id}. Answer with a single symbol. Do not print anything else."
    return sys_msg


def single_token_answer_prompt(question: str, options: List[str]):
    options_str = "\n".join([f"{option_id}. {answer}".strip() for option_id, answer in zip(option_ids, options)])
    user_prompt = f"Question: {question.strip()}\nOptions:\n{options_str}\n"
    return user_prompt


def single_token_answer_prompt_with_fallback_for_unknown_answers(question: str, options: List[str]):
    options_str = "\n".join([f"{option_id}. {answer}".strip() for option_id, answer in zip(option_ids, options)])
    user_prompt = f"Question: {question.strip()}\nOptions:\n{options_str}\n"
    return user_prompt


def single_token_answer_prompt_with_fallback_for_unknown_answers_alternative(question: str, options: List[str]):
    options_str = "\n".join([f"{option_id}. {answer}".strip() for option_id, answer in zip(option_ids, options)])
    user_prompt = f"Question: {question.strip()}\nOptions:\n{options_str}\n"
    return user_prompt
