import pandas as pd

answer_marker = ("[[", "]]")


def cot_sys_prompt_from_row(row: pd.Series):
    sys_msg = f"The following are grade school math word problems. Explain your thinking process step-by-step. At the end, write down the correct answer as a number by strictly following this format: {answer_marker[0]}correct_answer{answer_marker[1]}."
    return sys_msg


def cot_answer_prompt_from_row(row: pd.Series):
    assert isinstance(row["question"], str)
    return row["question"]
