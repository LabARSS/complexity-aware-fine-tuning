import pandas as pd

answer_marker = ("[[", "]]")


def cot_sys_prompt_from_row(row: pd.Series):
    sys_msg = f"The following are grade school math word problems. Explain your thinking process step-by-step. At the end, write down the correct answer as a number by strictly following this format: {answer_marker[0]}correct_answer{answer_marker[1]}."
    return sys_msg


def single_token_sys_prompt_from_row(row: pd.Series):
    sys_msg = f"The following are grade school math word problems. Write down only the correct answer and nothing else by strictly following this format: {answer_marker[0]}correct_answer{answer_marker[1]}."
    return sys_msg


def cot_answer_prompt_from_row(row: pd.Series) -> str:
    question = row["question"]
    assert isinstance(question, str)
    return question
