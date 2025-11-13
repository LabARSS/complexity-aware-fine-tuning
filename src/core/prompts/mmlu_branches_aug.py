from typing import List, Optional

option_ids = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


## DUPLICATED FROM mmlu_single_token_answer.py
# def single_token_sys_prompt(subject: str | None = None):
#     if subject is not None:
#         sys_msg = f"The following are multiple choice questions about {subject}."
#     else:
#         sys_msg = "The following are multiple choice questions."

#     sys_msg += " Choose a correct option letter. Answer with a single symbol. Do not print anything else."
#     return sys_msg

# def single_token_answer_prompt(question: str, options: List[str]):
#     options_str = "\n".join([f"{option_id}. {answer}".strip() for option_id, answer in zip(option_ids, options)])
#     user_prompt = f"Question: {question.strip()}\nOptions:\n{options_str}\n"
#     return user_prompt

def explain_sys_prompt(subject: Optional[str] = None) -> str:
    if subject:
        base = f"The following are multiple choice questions about {subject}."
    else:
        base = "The following are multiple choice questions."
    return base + " Given the correct option letter, explain why other options are wrong and the given option is correct."

def explain_user_prompt(question: str, options: List[str], gold_letter: str) -> str:
    opts = "\n".join([f"{oid}. {text}".strip() for oid, text in zip(option_ids, options)])
    return (
        f"Question: {question.strip()}\n"
        f"Options:\n{opts}\n"
        f"Correct option: {gold_letter}\n"
        f"Keep the explanation concise but specific."
    )

def error_review_sys_prompt(subject: Optional[str] = None) -> str:
    if subject:
        base = f"The following are multiple choice questions about {subject}."
    else:
        base = "The following are multiple choice questions."
    return base + " Given your previous answer and the correct option letter, analyze the error: explain why the chosen option is wrong and why the correct option is right."

def error_review_messages(question: str,
                          options: List[str],
                          model_letter: str,
                          gold_letter: str,
                          previous_reasoning: str) -> list[dict]:
    opts = "\n".join([f"{oid}. {text}".strip() for oid, text in zip(option_ids, options)])
    return [
        {"role": "user", "content": f"Question: {question.strip()}\nOptions:\n{opts}\n"},
        {"role": "assistant", "content": f"{previous_reasoning}\nAnswer: {model_letter}"},
        {"role": "user", "content": "Here is your previous reasoning:\n" + (previous_reasoning or "")},
        {"role": "user", "content": (
            f"Gold option: {gold_letter}.\n"
            f"Briefly explain why {gold_letter} is correct and why {model_letter} is wrong. "
            f"If your previous reasoning had mistakes, correct them. Be precise and concise."
        )},
    ]