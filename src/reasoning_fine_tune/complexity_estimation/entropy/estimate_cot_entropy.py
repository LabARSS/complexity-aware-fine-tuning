import gc
import json
import os

import pandas as pd
import torch
from tqdm import tqdm

from reasoning_fine_tune.complexity_estimation.entropy.logit_sequence_stats import collect_logit_sequence_stats
from reasoning_fine_tune.prompts.mmlu_cot_answer import answer_marker, cot_answer_prompt, cot_sys_prompt
from reasoning_fine_tune.utils.device import DEVICE
from reasoning_fine_tune.utils.embeddings import get_embeddings
from reasoning_fine_tune.utils.validation import validate_mmlu_answer


def estimate_dataset(
    in_filename,
    out_filename,
    model,
    tokenizer,
    get_subject_from_row,
    get_question_from_row,
    get_options_from_row,
    check_answer_correct,
    dump_every=1000,
    max_new_tokens=1024,
    get_sys_prompt=cot_sys_prompt,
    get_user_prompt=cot_answer_prompt,
):
    invalid_answers = 0

    if os.path.exists(out_filename):
        df = pd.read_parquet(
            out_filename,
        )
    else:
        df = pd.read_csv(
            in_filename,
            sep="\t",
            header=0,
        )

    model_name = model.config_class().model_type
    print(model_name)

    field_response = f"{model_name}_response"
    field_ans_token_index = f"{model_name}_ans_token_index"
    field_ans_correct = f"{model_name}_ans_correct"
    field_entropies_value = f"{model_name}_entropies"
    field_every_token_info = f"{model_name}_every_token_info"
    field_input_embeddings = f"{model_name}_input_embeddings"
    field_think_embeddings = f"{model_name}_think_embeddings"
    field_answer_embeddings = f"{model_name}_answer_embeddings"

    if field_ans_correct not in df.columns:
        df[field_ans_correct] = False
    if field_entropies_value not in df.columns:
        df[field_entropies_value] = ""
    if field_every_token_info not in df.columns:
        df[field_every_token_info] = ""
    if field_ans_token_index not in df.columns:
        df[field_ans_token_index] = -1
    if field_response not in df.columns:
        df[field_response] = ""
    if field_input_embeddings not in df.columns:
        df[field_input_embeddings] = ""
    if field_think_embeddings not in df.columns:
        df[field_think_embeddings] = ""
    if field_answer_embeddings not in df.columns:
        df[field_answer_embeddings] = ""

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if df.at[index, field_ans_token_index] != -1:
            continue

        gc.collect()
        if DEVICE == torch.device("cuda"):
            torch.cuda.empty_cache()

        # print(f"loop {index} -> start: {model.get_memory_footprint(return_buffers=True) / 10**9} GB")

        sys_prompt = get_sys_prompt(get_subject_from_row(row))
        user_prompt = get_user_prompt(get_question_from_row(row), get_options_from_row(row))
        # print(user_prompt)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            temperature=None,
            top_p=None,
            top_k=None,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        # print(f"loop {index} -> after generate: {model.get_memory_footprint(return_buffers=True) / 10**9} GB")

        input_length = inputs.input_ids.shape[1]
        response_raw = outputs.sequences[0, input_length:]
        response_decoded = tokenizer.decode(response_raw, skip_special_tokens=True)

        df.at[index, field_response] = response_decoded

        logit_stats = collect_logit_sequence_stats(outputs.scores)

        df.at[index, field_entropies_value] = json.dumps(logit_stats.entropies)
        df.at[index, field_every_token_info] = json.dumps(logit_stats.every_token_stats)

        output_str: str = ""
        answer_marker_start = -1
        answer_marker_end = -1
        for i, token in enumerate(logit_stats.greedy_tokens):
            token_str = tokenizer.decode(token)
            output_str += token_str

            if answer_marker_start == -1:
                if answer_marker[0] in output_str:
                    answer_marker_start = i
            elif answer_marker_end == -1:
                if answer_marker[1] in output_str:
                    answer_marker_end = i

        extracted_answer: str = ""
        if answer_marker_end != -1 and answer_marker_start != -1:
            ans_token_index = answer_marker_start + 1
            extracted_answer = tokenizer.decode(logit_stats.greedy_tokens[ans_token_index])
            df.at[index, field_ans_token_index] = ans_token_index

            answer_embeddings = get_embeddings(model, tokenizer, extracted_answer)
            if answer_embeddings is not None:
                df.at[index, field_answer_embeddings] = json.dumps(answer_embeddings)

            think_text = tokenizer.decode(logit_stats.greedy_tokens[:answer_marker_start])
            think_embeddings = get_embeddings(model, tokenizer, think_text)
            if think_embeddings is not None:
                df.at[index, field_think_embeddings] = json.dumps(think_embeddings)

        input_embeddings = get_embeddings(model, tokenizer, formatted_prompt)
        if input_embeddings is not None:
            df.at[index, field_input_embeddings] = json.dumps(input_embeddings)

        if validate_mmlu_answer(extracted_answer):
            # print(f"loop {index} -> after entropy: {model.get_memory_footprint(return_buffers=True) / 10**9} GB")
            df.at[index, field_ans_correct] = check_answer_correct(row, extracted_answer)
        else:
            invalid_answers += 1

        if index % dump_every == 0:
            df.to_parquet(out_filename, compression="gzip")

    df.to_parquet(out_filename, compression="gzip")
    print(f"Processed dataset {out_filename}. Total entries: {df.shape[0]}. Invalid answers: {invalid_answers}")
    return df
