import os
from concurrent import futures

import pandas as pd
from tqdm import tqdm

from reasoning_fine_tune.prompts.mmlu_cot_answer import answer_marker, cot_answer_prompt, cot_sys_prompt
from reasoning_fine_tune.utils.openrouter import openrouter
from reasoning_fine_tune.utils.validation import validate_mmlu_answer

chunk_size = 30


def call_remote_llm(args):
    try:
        sys_prompt, user_prompt, index, model, max_tokens = args

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion = openrouter.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
        return index, completion.choices[0].message.content
    except:
        return None


def distill_on_dataset(
    in_filename,
    out_filename,
    get_subject_from_row,
    get_question_from_row,
    get_options_from_row,
    check_answer_correct,
    dump_every=500,
    max_tokens=4096,
    model="deepseek/deepseek-chat-v3-0324",
    get_sys_prompt=cot_sys_prompt,
    get_user_prompt=cot_answer_prompt,
):
    invalid_answers = 0
    processed_rows = 0

    field_response = "distill_response"
    field_ans = "distill_answer"
    field_ans_correct = "distill_ans_correct"

    if os.path.exists(out_filename):
        print("Found an existing DF. Appending...")
        df = pd.read_csv(out_filename, sep="\t", dtype={field_response: "str", field_ans: "str"}, keep_default_na=False)
    else:
        df = pd.read_csv(
            in_filename,
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
        pooled_requests_args_list = []

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            if row[field_ans_correct]:
                continue

            processed_rows += 1

            if len(pooled_requests_args_list) < chunk_size:
                sys_prompt = get_sys_prompt(get_subject_from_row(row))
                user_prompt = get_user_prompt(get_question_from_row(row), get_options_from_row(row))
                pooled_requests_args_list.append((sys_prompt, user_prompt, index, model, max_tokens))

                if index != (df.shape[0] - 1):
                    continue

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

                if validate_mmlu_answer(extracted_answer):
                    df.at[index, field_ans] = extracted_answer
                    df.at[index, field_ans_correct] = check_answer_correct(df.iloc[index], extracted_answer)
                else:
                    invalid_answers += 1

                # print(
                #     f"response: {response}\nextracted_answer: {extracted_answer}\ncorrect:{df.at[index, field_ans_correct]}\n\n"
                # )

            if processed_rows % dump_every == 0:
                df.to_csv(out_filename, sep="\t", index=False)

    df.to_csv(out_filename, sep="\t", index=False)
    print(f"Processed dataset {out_filename}. Total entries: {df.shape[0]}. Invalid answers: {invalid_answers}")
    return df
