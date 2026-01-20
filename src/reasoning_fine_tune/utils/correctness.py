import pandas as pd

from reasoning_fine_tune.prompts.mmlu_single_token_answer import option_ids_w_fallback


def check_answer_correct_mmlu(row: pd.Series, model_answer: str):
    assert str(model_answer) in option_ids_w_fallback

    try:
        return int(row["answer_index"]) + 1 == int(model_answer.strip())
    except Exception:
        return False


def check_answer_correct_gsm8k(row: pd.Series, model_answer: str):
    assert isinstance(row["answer"], str)

    try:
        return row["answer"].lower() == model_answer.strip().lower()
    except Exception:
        return False


def check_answer_correct_arc(row: pd.Series, model_answer: str):
    assert isinstance(row["answerKey"], str)

    try:
        return row["answerKey"].lower() == model_answer.strip().lower()
    except Exception:
        return False


def check_answer_correct_gpqa(row: pd.Series, model_answer: str):
    assert isinstance(row["answer"], str)

    try:
        return row["answer"].lower() == model_answer.strip().lower()
    except Exception:
        return False


def check_answer_correct_medmcqa(row: pd.Series, model_answer: str):
    assert isinstance(row["answer"], str)

    try:
        return row["answer"].lower() == model_answer.strip().lower()
    except Exception:
        return False
