import re

import pandas as pd
from tqdm import tqdm

from reasoning_fine_tune.prompts.mmlu_single_token_answer import single_token_answer_prompt
from reasoning_fine_tune.prompts.oversmaple import (
    estimate_response_rating_system_prompt,
    oversample_system_prompt,
    valid_ratings,
)
from reasoning_fine_tune.utils.mistral_api_client import MistralAPIClient


def estimate_rating(api_client: MistralAPIClient, question, answer):
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
                {oversample_system_prompt}
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
    return rating_int


def paraphrase_with_model(api_client: MistralAPIClient, question):
    chat_response = api_client.query_model(
        [
            {
                "role": "system",
                "content": oversample_system_prompt,
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
    return response


def oversample_dataset(
    in_filename,
    out_filename,
    question_col: str,
    get_options_from_row,
    oversample_ratio: int,
    threshold=9,
    dump_every=100,
    model=None,
    sleep_duration= None
):
    in_df = pd.read_csv(
        in_filename,
        sep="\t",
        header=0,
    )
    out_df = pd.DataFrame(columns=in_df.columns)

    mistral_api_client = MistralAPIClient(model=model, sleep_duration=sleep_duration)

    invalid_entries = 0

    mistral_api_client.reset_api_limits()

    for index, row in tqdm(in_df.iterrows(), total=in_df.shape[0]):
        try:
            question = single_token_answer_prompt(row[question_col], get_options_from_row(row))

            for _ in range(oversample_ratio):
                try:
                    paraphrased_question = paraphrase_with_model(mistral_api_client, question)
                    mistral_api_client.wait()

                    rating = estimate_rating(
                        mistral_api_client,
                        question,
                        paraphrased_question,
                    )
                    if rating < threshold:
                        raise Exception("Response rating is too low")

                    new_row = row.copy()
                    new_row[question_col] = paraphrased_question
                    out_df = pd.concat([out_df, pd.DataFrame([new_row])], ignore_index=True)

                    mistral_api_client.wait()
                except:
                    print(f"Failed paraphrasing for question: {index}")
                    invalid_entries += 1

        except:
            print(f"Failed generating prompt for question: {index}")
            invalid_entries += oversample_ratio

        iterations = index * oversample_ratio
        if iterations % dump_every == 0:
            out_df.to_csv(out_filename, sep="\t", index=False)
            total_hits = 0
            for value in mistral_api_client.api_limit_hits_by_client_ids.values():
                total_hits += value
            print(f"Over {iterations} iterations we hit {total_hits} API limits")

    out_df.to_csv(out_filename, sep="\t", index=False)
    print(
        f"Processed dataset {out_filename}. Total entries: {in_df.shape[0] * oversample_ratio}. Invalid entries: {invalid_entries}."
    )
    return out_df
