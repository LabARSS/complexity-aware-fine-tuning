from pathlib import Path

import pandas as pd


def join_with_masj_education_levels(df, rating_threshold=9):
    masj_data_path = Path(__file__).parent.joinpath("../../../data/out/masj/mmlu_masj_education_levels.tsv")
    masj_edu_levels_df = pd.read_csv(
        masj_data_path,
        sep="\t",
        header=0,
    )

    masj_edu_levels_df = masj_edu_levels_df[masj_edu_levels_df["masj_rating"] >= rating_threshold]

    merged_df = pd.merge(df, masj_edu_levels_df[["question_id", "masj_complexity"]], how="inner", on="question_id")

    merged_df.rename(columns={"masj_complexity": "masj_edu_level"}, inplace=True)

    merged_df["masj_edu_level_norm"] = 0.0
    merged_df.loc[merged_df["masj_edu_level"] == "high_school_and_easier", "masj_edu_level_norm"] = 0.2
    merged_df.loc[merged_df["masj_edu_level"] == "undergraduate", "masj_edu_level_norm"] = 0.4
    merged_df.loc[merged_df["masj_edu_level"] == "graduate", "masj_edu_level_norm"] = 0.6
    merged_df.loc[merged_df["masj_edu_level"] == "postgraduate", "masj_edu_level_norm"] = 0.8
    # Ensure the target variable is in the correct format
    merged_df["masj_edu_level_norm"] = merged_df["masj_edu_level_norm"].astype(float)

    return merged_df
