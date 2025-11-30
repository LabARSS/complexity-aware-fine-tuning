import math
import os
from concurrent import futures

import pandas as pd
from tqdm import tqdm

from core.prompts.mmlu_single_token_answer import single_token_answer_prompt, single_token_sys_prompt
from core.utils.chunker import chunker
from core.utils.openrouter import openrouter
from core.utils.validation import validate_mmlu_answer

chunk_size = 30


def call_remote_llm(args):
    try:
        sys_prompt, user_prompt, index, model, max_tokens, provider = args

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion = openrouter.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, logprobs=True, top_logprobs=20, extra_body={"provider": {"order": [provider]}}
        )

        return (
            index,
            completion.choices[0].message.content,
            completion.choices[0].logprobs.content[0].logprob,
            [
                {"token": logprob.token, "logprob": logprob.logprob}
                for logprob in completion.choices[0].logprobs.content[0].top_logprobs
            ],
        )
    except:
        return None


def distill_logprobs(
    in_filename,
    out_filename,
    model,
    provider,
    get_subject_from_row,
    get_question_from_row,
    get_options_from_row,
    check_answer_correct,
    dump_every=10,
    max_tokens=1,
    get_sys_prompt=single_token_sys_prompt,
    get_user_prompt=single_token_answer_prompt,
):
    invalid_answers = 0

    field_logprobs = "field_logprobs"
    field_ans = "distill_answer"
    field_ans_correct = "distill_ans_correct"
    field_ans_prob = "distill_answer_prob"
    field_entropy = "distill_entropy"

    if os.path.exists(out_filename):
        df = pd.read_parquet(out_filename)
    else:
        df = pd.read_csv(
            in_filename,
            sep="\t",
        )

    # print(df.dtypes)

    if field_ans_correct not in df.columns:
        df[field_ans_correct] = False
    if field_logprobs not in df.columns:
        df[field_logprobs] = pd.Series([[]] * len(df), dtype=object)
    if field_ans not in df.columns:
        df[field_ans] = ""
    if field_entropy not in df.columns:
        df[field_entropy] = 0.0
    if field_ans_prob not in df.columns:
        df[field_ans_prob] = 0.0

    first_chunk = True

    with futures.ThreadPoolExecutor(max_workers=chunk_size) as pool:
        for chunk_idx, chunk in tqdm(enumerate(chunker(df, chunk_size)), total=int(df.shape[0] / chunk_size)):
            args_list = []

            for index, row in chunk.iterrows():
                if df.at[index, field_ans_correct] == True:
                    continue

                sys_prompt = get_sys_prompt(get_subject_from_row(row))
                user_prompt = get_user_prompt(get_question_from_row(row), get_options_from_row(row))
                args_list.append((sys_prompt, user_prompt, index, model, max_tokens, provider))

            results = list(pool.map(call_remote_llm, args_list))

            for result in results:
                if result is None:
                    invalid_answers += 1
                    continue

                index, response, response_logprob, logprobs = result

                df.at[index, field_logprobs] = logprobs

                if validate_mmlu_answer(response):
                    df.at[index, field_ans] = response
                    df.at[index, field_ans_correct] = check_answer_correct(df.iloc[index], response)

                    df.at[index, field_ans_prob] = math.exp(response_logprob)

                    logprobs_num: list[float] = [lp["logprob"] for lp in logprobs]
                    entropy = sum([p * math.exp(p) for p in logprobs_num])
                    df.at[index, field_entropy] = -entropy
                else:
                    invalid_answers += 1

                if first_chunk:
                    print(
                        f"index: {index}\nresponse: {response}\nextracted_answer: {df.at[index, field_ans]}\ncorrect:{df.at[index, field_ans_correct]}\nanswer probability: {df.at[index, field_ans_prob]}\nlogprobs:{df.at[index, field_logprobs]}\n\n"
                    )
                    first_chunk = False

                if chunk_idx % dump_every == 0:
                    df.to_parquet(out_filename, index=False)

    df.to_parquet(out_filename, index=False)
    print(f"Processed dataset {out_filename}. Total entries: {df.shape[0]}. Invalid answers: {invalid_answers}")
    return df
