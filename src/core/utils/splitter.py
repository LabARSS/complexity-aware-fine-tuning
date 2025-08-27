import numpy as np
from pandas import DataFrame

from core.utils.validation import keep_only_valid_and_known_answers


def split_into_even_chunks(
    df: DataFrame, split_by_col_name: str, answer_col_name: str, chunk_cnt: int = 5
) -> list[DataFrame]:
    filtered_df = keep_only_valid_and_known_answers(df, answer_col_name)

    sorted_df = filtered_df.sort_values(split_by_col_name, ascending=True)

    chunk_len = len(sorted_df) // chunk_cnt

    chunks: list[DataFrame] = []
    for i in range(chunk_cnt):
        start_idx = i * chunk_len
        # Python (and pandas for that matter) is OK with end index to be out of bounds
        end_idx = start_idx + chunk_len
        chunk = sorted_df.iloc[start_idx:end_idx]
        chunk.reset_index(drop=True, inplace=True)
        chunks.append(chunk)

    return chunks


def split_chunk_into_train_test(chunk: DataFrame, test_allocation: float):
    chunk_len = len(chunk)
    test_idx = np.random.choice(chunk_len, round(chunk_len * test_allocation), replace=False)

    test_df = chunk.iloc[test_idx]
    train_df = chunk.drop(test_idx)

    return train_df, test_df
