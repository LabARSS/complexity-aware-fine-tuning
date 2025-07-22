import os.path
import re

import pandas as pd
from tqdm import tqdm

from reasoning_fine_tune.prompts.estimate_education_level import (
    estimate_education_level_system_prompt,
    estimate_response_rating_system_prompt,
    valid_education_levels,
    valid_ratings,
)
from reasoning_fine_tune.prompts.mmlu_single_token_answer import single_token_answer_prompt
from reasoning_fine_tune.utils.mistral_api_client import MistralAPIClient

FIELD_RATING = "masj_rating"
FIELD_COMPLEXITY = "masj_complexity"


def estimate_rating(api_client: MistralAPIClient, df, index, question, answer):
    chat_response = api_client.query_model(
        [
            {
                "role": "system",
                "content": estimate_response_rating_system_prompt,
            },
            {
                "role": "user",
                "content": f"""
                [Instructions for Assistant]
                {estimate_education_level_system_prompt}
                [End of Instructions for Assistant]

                [Question]
                {question}
                [End of Question]

                [The Start of Assistant’s Answer]
                {answer}
                [The End of Assistant’s Answer]

                You must rate the assistant's response on a scale of 1 to 10 (where 10 is the best and 1 is the worst) by strictly following this format: "[[rating]]", for example:"Rating: [[6]]"
                """,
            },
        ]
    )
    response = chat_response.choices[0].message.content
    # print(response)

    rating = re.search("\\[\\[(\\d+?)\\]\\]", response).group(1)
    # print(rating)
    rating_int = int(rating)
    assert rating_int in valid_ratings
    df.at[index, FIELD_RATING] = rating_int


def estimate_education_level_with_model(api_client: MistralAPIClient, df, index, question):
    chat_response = api_client.query_model(
        [
            {
                "role": "system",
                "content": estimate_education_level_system_prompt,
            },
            {
                "role": "user",
                "content": f"""
                [Question Start]
                {question}
                [Question End]
                """,
            },
        ]
    )
    response = chat_response.choices[0].message.content

    complexity_str = re.search("\\[\\[(.+?)\\]\\]", response).group(1)
    assert complexity_str in valid_education_levels
    df.at[index, FIELD_COMPLEXITY] = complexity_str

    return response


def estimate_dataset(
    in_filename,
    out_filename,
    get_question_from_row,
    get_options_from_row,
    dump_every=100,
):
    if os.path.exists(out_filename):
        in_filename = out_filename

    df = pd.read_csv(
        in_filename,
        sep="\t",
        header=0,
    )

    mistral_api_client = MistralAPIClient()

    invalid_entries = 0

    mistral_api_client.reset_api_limits()

    if FIELD_COMPLEXITY not in df.columns:
        df[FIELD_COMPLEXITY] = pd.Series(dtype="str")
    if FIELD_RATING not in df.columns:
        df[FIELD_RATING] = pd.Series(dtype="int")
        df[FIELD_RATING] = 0

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if df.at[index, FIELD_COMPLEXITY] in valid_education_levels and df.at[index, FIELD_RATING] in valid_ratings:
            continue

        question = single_token_answer_prompt(get_question_from_row(row), get_options_from_row(row))
        # print(complexity_user_prompt)

        try:
            response_complexity = estimate_education_level_with_model(mistral_api_client, df, index, question)
        except:
            response_complexity = None
            invalid_entries += 1

        mistral_api_client.wait()

        if response_complexity is not None:
            try:
                estimate_rating(
                    mistral_api_client,
                    df,
                    index,
                    question,
                    response_complexity,
                )
                mistral_api_client.wait()
            except:
                invalid_entries += 1

        if index % dump_every == 0:
            df.to_csv(out_filename, sep="\t", index=False)
            total_hits = 0
            for value in mistral_api_client.api_limit_hits_by_client_ids.values():
                total_hits += value
            print(f"Over {index} iterations we hit {total_hits} API limits")

    df.to_csv(out_filename, sep="\t", index=False)
    print(f"Processed dataset {out_filename}. Total entries: {df.shape[0]}. Invalid entries: {invalid_entries}.")
    return df
