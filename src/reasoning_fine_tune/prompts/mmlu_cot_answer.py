from typing import List

from reasoning_fine_tune.prompts.mmlu_option_ids import fallback_option_id, option_ids

answer_marker = ("[[", "]]")


def cot_sys_prompt(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += f" Explain your thinking process step-by-step. At the end, choose a correct option by strictly following this format: {answer_marker[0]}correct_option{answer_marker[1]}."
    return sys_msg


def cot_sys_prompt_with_fallback_for_unknown_answers(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += f" Explain your thinking process step-by-step. At the end, if you are certain about the answer choose a correct option by strictly following this format: {answer_marker[0]}correct_option{answer_marker[1]}, otherwise return {answer_marker[0]}{fallback_option_id}{answer_marker[1]}."
    return sys_msg


def cot_sys_prompt_with_fallback_for_unknown_answers_alternative(subject: str | None = None):
    if subject is not None:
        sys_msg = f"The following are multiple choice questions about {subject}."
    else:
        sys_msg = "The following are multiple choice questions."

    sys_msg += f" Explain your thinking process step-by-step. At the end, if you know the answer choose a correct option by strictly following this format: {answer_marker[0]}correct_option{answer_marker[1]}, otherwise return {answer_marker[0]}{fallback_option_id}{answer_marker[1]}."
    return sys_msg


def cot_answer_prompt(question: str, options: List[str]):
    options_str = "\n".join([f"{option_id}. {answer}".strip() for option_id, answer in zip(option_ids, options)])
    user_prompt = f"Question: {question.strip()}\nOptions:\n{options_str}\n"
    return user_prompt
