import gc
import os

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding

import core.prompts.mmlu_single_token_answer as mmlu_prompts
from core.complexity_estimation.entropy.logit_entropy import compute_entropy_from_logits
from core.utils.device import DEVICE, move_batch_to_device
from core.utils.seed import set_seed
from core.utils.validation import validate_mmlu_answer

field_entropy_value = "entropy_value"


def add_single_token_entropy_cols(df):
    if field_entropy_value not in df.columns:
        df[field_entropy_value] = 0.0


def process_single_token_entropy_response(df, row_idx, inputs, outputs, response_idx, response, model, tokenizer):
    final_token_logits = outputs.scores[-1][response_idx]
    entropy = compute_entropy_from_logits(final_token_logits)

    df.at[row_idx, field_entropy_value] = entropy

    return response


def estimate_dataset(
    in_filename,
    out_filename,
    model,
    tokenizer,
    get_subject_from_row,
    get_question_from_row,
    get_options_from_row,
    check_answer_correct,
    max_new_tokens=1,
    get_sys_prompt=mmlu_prompts.single_token_sys_prompt,
    get_user_prompt=mmlu_prompts.single_token_answer_prompt,
    batch_size=16,
    dump_every=1000,
    add_columns=add_single_token_entropy_cols,
    process_response=process_single_token_entropy_response,
):
    invalid_formatting = 0
    correct_answers = 0

    set_seed()

    if os.path.exists(out_filename):
        in_filename = out_filename

    if in_filename.endswith(".csv"):
        df = pd.read_csv(
            in_filename,
            sep="\t",
            header=0,
        )
    else:
        df = pd.read_parquet(
            in_filename,
        )

    model_name = model.config_class().model_type
    print(model_name)

    field_response = "entropy_response"
    if field_response not in df.columns:
        df[field_response] = ""
    field_ans_correct = "entropy_ans_correct"
    if field_ans_correct not in df.columns:
        df[field_ans_correct] = False

    add_columns(df)

    tokenizer.padding_side = "left"

    ds = Dataset.from_pandas(df)

    print(f"\nDs len = {len(ds)}\n")

    def preprocess_ds(row):
        sys_prompt = get_sys_prompt(get_subject_from_row(row))
        user_prompt = get_user_prompt(get_question_from_row(row), get_options_from_row(row))
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tokenized = tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        for k in tokenized:
            tokenized[k] = tokenized[k].squeeze(0)
        return tokenized

    ds = ds.map(
        preprocess_ds,
        num_proc=4,
        batched=False,
        remove_columns=[col for col in ds.column_names if col not in ["input_ids", "attention_mask"]],
    )

    print("\nDs sample:\n")
    print("\n\n".join(tokenizer.batch_decode(ds[:3]["input_ids"])))

    data_collator = DataCollatorWithPadding(tokenizer)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    batch = next(iter(dataloader))

    pbar = tqdm(dataloader)
    for batch_idx, batch in enumerate(pbar):
        last_row_in_batch_idx = batch_idx * batch_size + batch_size - 1
        if df.at[last_row_in_batch_idx, field_response] != "":
            continue

        gc.collect()
        if DEVICE == torch.device("cuda"):
            torch.cuda.empty_cache()

        batch = move_batch_to_device(batch, DEVICE)

        outputs = model.generate(
            **batch,
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

        # They are all padded to the same length
        input_length = batch["input_ids"].shape[1]
        response_token_batch = outputs.sequences[:, input_length:]
        response_batch = tokenizer.batch_decode(response_token_batch, skip_special_tokens=True)

        for response_idx, response in enumerate(response_batch):
            row_idx = batch_idx * batch_size + response_idx
            df.at[row_idx, field_response] = response

            answer = process_response(df, row_idx, batch, outputs, response_idx, response, model, tokenizer)

            if validate_mmlu_answer(answer):
                is_correct = check_answer_correct(df.iloc[row_idx], answer)
                df.at[row_idx, field_ans_correct] = is_correct
                if is_correct:
                    correct_answers += 1
            else:
                invalid_formatting += 1

        total = batch_idx * batch_size + len(response_batch)
        if total % dump_every == 0:
            df.to_parquet(out_filename, compression="gzip")
        pbar.set_description(f"accuracy={correct_answers / total:.2f} / invalid formatting={invalid_formatting}")

    df.to_parquet(out_filename, compression="gzip")
    print(f"Processed dataset {out_filename}. Total entries: {df.shape[0]}. Invalid formatting: {invalid_formatting}")
    return df
