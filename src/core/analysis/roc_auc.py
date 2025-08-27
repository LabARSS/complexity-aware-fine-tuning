from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import roc_auc_score


def normalize_score(column, norm_basis: float | None = None):
    norm_basis = norm_basis or column.max()
    return 1 - column / norm_basis


@dataclass
class ROC_AUC:
    roc_auc: float
    gini: float


def calculate_roc_auc(y_true, y_score):
    roc_auc = roc_auc_score(y_true, y_score)
    gini = abs(2 * roc_auc - 1)
    return ROC_AUC(roc_auc=roc_auc, gini=gini)


def calculate_accuracy(df, model_answer_correct_col):
    return df[model_answer_correct_col].sum() / len(df)


def calculate_roc_auc_by_category(
    df, model_name, category_cols, model_answer_correct_col, score_col, threshold=10, norm_basis: float | None = None
):
    results = []

    roc_auc = calculate_roc_auc(df[model_answer_correct_col], normalize_score(df[score_col], norm_basis))

    results.append(
        {
            "category": "ALL",
            "roc_auc": roc_auc.roc_auc,
            "accuracy": calculate_accuracy(df, model_answer_correct_col),
            "gini": roc_auc.gini,
            "num_samples": len(df),
            "model": model_name,
            "metric": score_col,
        }
    )

    for category_col in category_cols:
        for category in df[category_col].unique():
            df_cat = df[df[category_col] == category]
            if len(df_cat) < threshold:
                continue

            roc_auc = calculate_roc_auc(df_cat[model_answer_correct_col], normalize_score(df_cat[score_col]))

            results.append(
                {
                    "category": category,
                    "roc_auc": roc_auc.roc_auc,
                    "accuracy": calculate_accuracy(df_cat, model_answer_correct_col),
                    "gini": roc_auc.gini,
                    "num_samples": len(df_cat),
                    "model": model_name,
                    "metric": score_col,
                }
            )

    return pd.DataFrame(results)
