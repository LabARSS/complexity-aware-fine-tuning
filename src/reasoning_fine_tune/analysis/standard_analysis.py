import pandas as pd
from IPython.display import display
from transformers import AutoTokenizer

from reasoning_fine_tune.analysis.join_with_masj_education_levels import join_with_masj_education_levels
from reasoning_fine_tune.analysis.join_with_masj_reasoning_score import join_with_masj_reasoning_score
from reasoning_fine_tune.analysis.roc_auc import calculate_roc_auc_by_category
from reasoning_fine_tune.analysis.visualize_all import visualize_all
from reasoning_fine_tune.utils.processing import extract_cot_answer_entropy_from_row, extract_cot_answer_from_row
from reasoning_fine_tune.utils.validation import keep_only_valid_and_known_answers


def standard_analysis_single_token_response(
    df_path, ans_col, ans_correct_col, entropy_col, title: str, show=["graphs", "tables"]
):
    df = pd.read_csv(
        df_path,
        sep="\t",
        header=0,
        dtype={ans_col: "str"},
    )

    print(df.value_counts(ans_col, dropna=False))
    df = keep_only_valid_and_known_answers(df, ans_col)
    print(df.value_counts(ans_col, dropna=False))

    df = join_with_masj_education_levels(df)
    df = join_with_masj_reasoning_score(df)

    save_to_dir = f"single token/{title.lower()}"

    if "graphs" in show:
        visualize_all(df, entropy_col, ans_correct_col, model_name=title, save_to=f"{save_to_dir}/entropy")

        visualize_all(
            df,
            "masj_edu_level_norm",
            ans_correct_col,
            model_name=title,
            x_label="Education level",
            save_to=f"{save_to_dir}/edu_level",
        )

        visualize_all(
            df,
            "masj_num_reasoning_steps_norm",
            ans_correct_col,
            model_name=title,
            x_label="Reasoning score",
            save_to=f"{save_to_dir}/reasoning_score",
        )

    if "tables" in show:
        roc_auc_entropy = calculate_roc_auc_by_category(
            df,
            category_cols=["category", "masj_edu_level", "masj_num_reasoning_steps"],
            model_answer_correct_col=ans_correct_col,
            score_col=entropy_col,
            model_name=title,
        )
        display(roc_auc_entropy)

        roc_auc_edu_level = calculate_roc_auc_by_category(
            df,
            category_cols=[],
            model_answer_correct_col=ans_correct_col,
            score_col="masj_edu_level_norm",
            model_name=title,
            norm_basis=1.0,
        )
        display(roc_auc_edu_level)

        roc_auc_reasoning_score = calculate_roc_auc_by_category(
            df,
            category_cols=[],
            model_answer_correct_col=ans_correct_col,
            score_col="masj_num_reasoning_steps_norm",
            model_name=title,
            norm_basis=1.0,
        )
        display(roc_auc_reasoning_score)


def standard_analysis_cot_response(
    df_path,
    col_ans_token_index,
    col_every_token_info,
    col_entropies,
    col_ans_correct,
    model_name,
    title,
    show=["graphs", "tables"],
):
    df = pd.read_parquet(df_path)

    print(df.value_counts(col_ans_token_index, dropna=False))

    len_before = len(df)
    # Filter out unanswered questions (col_ans_token_index == -1)
    df = df[df[col_ans_token_index] != -1]
    print(f"Len = {len_before} before filtering and {len(df)} after filtering")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df["model_answer"] = df.apply(
        lambda row: extract_cot_answer_from_row(tokenizer, row, col_every_token_info, col_ans_token_index), axis=1
    )
    df["model_answer_entropy"] = df.apply(
        lambda row: extract_cot_answer_entropy_from_row(row, col_entropies, col_ans_token_index), axis=1
    )

    # Filter out incorrectly formatted answers (answer token is not one of the options)
    print(df.value_counts("model_answer", dropna=False))
    df = keep_only_valid_and_known_answers(df, "model_answer")
    print(df.value_counts("model_answer", dropna=False))

    df = join_with_masj_education_levels(df)
    df = join_with_masj_reasoning_score(df)

    save_to_dir = f"cot/{title.lower()}"

    if "graphs" in show:
        visualize_all(df, "model_answer_entropy", col_ans_correct, model_name=title, save_to=f"{save_to_dir}/entropy")

        visualize_all(
            df,
            "masj_edu_level_norm",
            col_ans_correct,
            model_name=title,
            x_label="Education level",
            save_to=f"{save_to_dir}/edu_level",
        )

        visualize_all(
            df,
            "masj_num_reasoning_steps_norm",
            col_ans_correct,
            model_name=title,
            x_label="Reasoning score",
            save_to=f"{save_to_dir}/reasoning_score",
        )

    if "tables" in show:
        roc_auc_entropy = calculate_roc_auc_by_category(
            df,
            category_cols=["category", "masj_edu_level", "masj_num_reasoning_steps"],
            model_answer_correct_col=col_ans_correct,
            score_col="model_answer_entropy",
            model_name=title,
        )
        display(roc_auc_entropy)

        roc_auc_edu_level = calculate_roc_auc_by_category(
            df,
            category_cols=[],
            model_answer_correct_col=col_ans_correct,
            score_col="masj_edu_level_norm",
            model_name=title,
            norm_basis=1.0,
        )
        display(roc_auc_edu_level)

        roc_auc_reasoning_score = calculate_roc_auc_by_category(
            df,
            category_cols=[],
            model_answer_correct_col=col_ans_correct,
            score_col="masj_num_reasoning_steps_norm",
            model_name=title,
            norm_basis=1.0,
        )
        display(roc_auc_reasoning_score)
