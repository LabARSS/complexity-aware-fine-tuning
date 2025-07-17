from pandas import DataFrame

from reasoning_fine_tune.utils.validation import keep_only_valid_and_known_answers


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
        chunks.append(chunk)

    return chunks
