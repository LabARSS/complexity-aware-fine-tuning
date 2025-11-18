import os
from concurrent import futures
from typing import Callable, cast

import pandas as pd
from pydraconf import PydraConfig
from tqdm import tqdm

from reasoning_fine_tune.prompts.mmlu_cot_answer import answer_marker
from reasoning_fine_tune.utils.openrouter import openrouter

chunk_size = 20


def call_remote_llm(args: tuple[str, str, int, str, int]) -> tuple[int, str] | None:
    sys_prompt, user_prompt, index, model, max_tokens = args
    try:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion = openrouter.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
        return index, completion.choices[0].message.content
    except Exception as e:
        print(f"call_remote_llm: error processing index {index}: {e}")
        return None


class DistillConfig(PydraConfig):
    in_filename: str
    out_filename: str
    check_answer_correct: Callable[[pd.Series, str], bool]
    model: str
    dump_every: int = 500
    max_tokens: int = 16384
    get_sys_prompt: Callable[[pd.Series], str]
    get_user_prompt: Callable[[pd.Series], str]


def distill_on_dataset(
    config: DistillConfig,
):
    invalid_answers = 0
    processed_rows = 0

    field_response = "distill_response"
    field_ans = "distill_answer"
    field_ans_correct = "distill_ans_correct"

    file_type = os.path.splitext(config.in_filename)[1]

    if file_type == ".jsonl":
        df = pd.read_json(config.out_filename, lines=True)
    else:
        if os.path.exists(config.out_filename):
            print("Found an existing DF. Appending...")
            df = pd.read_csv(
                config.out_filename, sep="\t", dtype={field_response: "str", field_ans: "str"}, keep_default_na=False
            )
        else:
            df = pd.read_csv(
                config.in_filename,
                sep="\t",
            )
    # print(df.dtypes)

    if field_ans_correct not in df.columns:
        df[field_ans_correct] = False
    if field_response not in df.columns:
        df[field_response] = ""
    if field_response not in df.columns:
        df[field_ans] = ""

    with futures.ThreadPoolExecutor(max_workers=chunk_size) as pool:
        pooled_requests_args_list: list[tuple[str, str, int, str, int]] = []

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            if row[field_ans_correct]:
                continue

            processed_rows += 1

            if len(pooled_requests_args_list) < chunk_size:
                sys_prompt = config.get_sys_prompt(row)
                user_prompt = config.get_user_prompt(row)
                pooled_requests_args_list.append(
                    (sys_prompt, user_prompt, cast(int, index), config.model, config.max_tokens)
                )

                if index != (df.shape[0] - 1):
                    continue

            # print("Calling API...")
            results = list(pool.map(call_remote_llm, pooled_requests_args_list))
            pooled_requests_args_list = []

            for result in results:
                if result is None:
                    invalid_answers += 1
                    continue

                index, response = result

                df.at[index, field_response] = response

                answer_marker_start = response.find(answer_marker[0])
                answer_marker_end = response.find(answer_marker[1])

                extracted_answer = ""
                if answer_marker_end != -1 and answer_marker_start != -1:
                    extracted_answer = response[answer_marker_start + len(answer_marker[0]) : answer_marker_end]

                try:
                    df.at[index, field_ans] = extracted_answer
                    df.at[index, field_ans_correct] = config.check_answer_correct(df.iloc[index], extracted_answer)
                except Exception:
                    invalid_answers += 1

                # print(
                #     f"response: {response}\nextracted_answer: {extracted_answer}\ncorrect:{df.at[index, field_ans_correct]}\n\n"
                # )

            if processed_rows % config.dump_every == 0:
                if file_type == ".jsonl":
                    df.to_json(config.out_filename, lines=True, orient="records")
                else:
                    df.to_csv(config.out_filename, sep="\t", index=False)

    if file_type == ".jsonl":
        df.to_json(config.out_filename, lines=True, orient="records")
    else:
        df.to_csv(config.out_filename, sep="\t", index=False)
    print(f"Processed dataset {config.out_filename}. Total entries: {df.shape[0]}. Invalid answers: {invalid_answers}")
    return df
