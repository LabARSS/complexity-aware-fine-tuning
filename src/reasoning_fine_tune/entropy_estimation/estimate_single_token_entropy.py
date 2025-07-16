import gc

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding

import reasoning_fine_tune.prompts.mmlu_single_token_answer as mmlu_prompts
from reasoning_fine_tune.entropy_estimation.logit_entropy import compute_entropy_from_logits
from reasoning_fine_tune.utils.device import DEVICE, move_batch_to_device
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
    max_new_tokens=1,
    get_sys_prompt=mmlu_prompts.single_token_sys_prompt,
    get_user_prompt=mmlu_prompts.single_token_answer_prompt,
    batch_size=16,
):
    invalid_answers = 0

    df = pd.read_csv(
        in_filename,
        sep="\t",
        header=0,
    )

    model_name = model.config_class().model_type
    print(model_name)

    field_ans = f"entropy_ans_{model_name}"
    field_ans_correct = f"entropy_ans_correct_{model_name}"
    field_entropy_value = f"entropy_value_{model_name}"

    if field_ans_correct not in df.columns:
        df[field_ans_correct] = False
    if field_entropy_value not in df.columns:
        df[field_entropy_value] = 0.0
    if field_ans not in df.columns:
        df[field_ans] = ""

    tokenizer.padding_side = "right"

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
        remove_columns=[col for col in ds.column_names if col not in ["input_ids", "attention_mask", "label"]],
    )

    print("\nDs sample:\n")
    print("\n\n".join(tokenizer.batch_decode(ds[:3]["input_ids"])))

    data_collator = DataCollatorWithPadding(tokenizer)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

    batch = next(iter(dataloader))

    for batch_idx, batch in enumerate(tqdm(dataloader)):
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
        input_length = batch["input_ids"][0].shape[1]
        answer_token_batch = outputs.sequences[:, input_length:]
        answer_batch = tokenizer.decode(answer_token_batch, skip_special_tokens=True)

        for answer_idx, answer in enumerate(answer_batch):
            row_idx = batch_idx * batch_size + answer_idx
            df.at[row_idx, field_ans] = answer
            # generated token position, batch_dim
            final_token_logits = outputs.scores[-1][0]
            entropy = compute_entropy_from_logits(final_token_logits)
            df.at[row_idx, field_entropy_value] = entropy

            if validate_mmlu_answer(answer):
                df.at[row_idx, field_ans_correct] = check_answer_correct(df.iloc[row_idx], answer)
            else:
                invalid_answers += 1

        # print(
        #     f"Answer: {answer}\nEntropy: {df.at[index, field_entropy_value]}\nis_correct: {df.at[index, field_ans_correct]}\ndims:{input_length}, {outputs.sequences.shape}\n\n"
        # )

    df.to_csv(out_filename, sep="\t", index=False)
    print(f"Processed dataset {out_filename}. Total entries: {df.shape[0]}. Invalid answers: {invalid_answers}")
    return df
