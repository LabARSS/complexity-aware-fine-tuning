import gc

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

def estimate_dataset(
    in_filename,
    out_filename,
    model,
    tokenizer,
    get_subject_from_row,
    get_question_from_row,
    get_options_from_row,
    get_sys_prompt=mmlu_prompts.single_token_sys_prompt,
    get_user_prompt=mmlu_prompts.single_token_answer_prompt,
    batch_size=16,
):
    invalid_formatting = 0
    correct_answers = 0

    set_seed()

    df = pd.read_csv(
        in_filename,
        sep="\t",
        header=0,
    )

    model_name = model.config_class().model_type
    print(model_name)

    field_ans = f"cross_entropy_ans_{model_name}"
    field_ans_correct = f"cross_entropy_ans_correct_{model_name}"
    field_ce_value = f"cross_entropy_value_{model_name}"

    if field_ans_correct not in df.columns:
        df[field_ans_correct] = False
    if field_ce_value not in df.columns:
        df[field_ce_value] = 0.0
    if field_ans not in df.columns:
        df[field_ans] = ""

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
            
        # Will use lower register for true answer
        correct_letter = row["answer"].strip().lower() 
        correct_token_ids = tokenizer.encode(correct_letter, add_special_tokens=False)
        
        if len(correct_token_ids) != 1:
            print(f"Warning: correct letter '{correct_letter}' is tokenized into {len(correct_token_ids)} tokens: {correct_token_ids}")
            correct_token_id = correct_token_ids[0] if correct_token_ids else -1
        else:
            correct_token_id = correct_token_ids[0]
            
        tokenized["correct_token_id"] = torch.tensor(correct_token_id, dtype=torch.long)
        
        return tokenized

    ds = ds.map(
        preprocess_ds,
        num_proc=4,
        batched=False,
        remove_columns=[col for col in ds.column_names if col not in ["input_ids", "attention_mask", "correct_token_id"]],
    )

    print("\nDs sample:\n")
    print("\n\n".join(tokenizer.batch_decode(ds[:3]["input_ids"])))

    data_collator = DataCollatorWithPadding(tokenizer)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    batch = next(iter(dataloader))

    pbar = tqdm(dataloader)
    for batch_idx, batch in enumerate(pbar):
        gc.collect()
        if DEVICE == torch.device("cuda"):
            torch.cuda.empty_cache()

        # save before pop
        correct_token_ids = batch.pop("correct_token_id")

        batch = move_batch_to_device(batch, DEVICE)
        correct_token_ids = correct_token_ids.to(DEVICE)

        with torch.no_grad():
            with torch.autocast(device_type="cuda" if "cuda" in str(DEVICE) else "cpu", 
                                dtype=torch.bfloat16 if DEVICE == torch.device("cuda") else torch.float32):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=True
                )
        
        last_token_logits = outputs.logits[:, -1,:]  # [batch_size, vocab_size]
        
        generated_token_ids = torch.argmax(last_token_logits, dim=-1)
        answer_batch = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)

        for answer_idx, answer in enumerate(answer_batch):
            row_idx = batch_idx * batch_size + answer_idx
            df.at[row_idx, field_ans] = answer

            example_logits = last_token_logits[answer_idx]
            correct_token_id = correct_token_ids[answer_idx]

            # Calculation of cross entropy
            log_softmax = torch.log_softmax(example_logits, dim=-1)
            cross_entropy = -log_softmax[correct_token_id].item()
            df.at[row_idx, field_ce_value] = cross_entropy

            # Compare without register
            normalized_answer = answer.strip().lower()
            is_correct_format = validate_mmlu_answer(normalized_answer)

            if is_correct_format:
                # Compare in lower register
                correct_letter = str(df.iloc[row_idx]["answer"]).strip().lower()
                is_correct = normalized_answer == correct_letter
                df.at[row_idx, field_ans_correct] = is_correct
                if is_correct:
                    correct_answers += 1
            else:
                invalid_formatting += 1

            # DEBUGGING:
            if batch_idx == 0 and answer_idx < 5:
                probs = torch.softmax(example_logits.float(), dim=-1)
                topk_probs, topk_ids = torch.topk(probs, 5)
                topk_tokens = tokenizer.batch_decode(topk_ids, skip_special_tokens=True)

                print(f"\nSample {answer_idx}:")
                print(f"Correct token: {tokenizer.decode(correct_token_id)} (id={correct_token_id})")
                print(f"Generated answer: {answer} (normalized: {normalized_answer})")
                print(f"Cross entropy: {cross_entropy:.4f}")
                print(f"Correct token probability: {probs[correct_token_id].item():.6f}")
                print("Top 5 predictions:")
                for i in range(5):
                    print(f"  {topk_tokens[i]} ({topk_ids[i].item()}): {topk_probs[i].item():.6f}")
                print(f"Is correct: {is_correct} | Format valid: {is_correct_format}")


    total = batch_idx * batch_size + len(answer_batch)
    pbar.set_description(f"accuracy={correct_answers / total:.2f} / invalid formatting={invalid_formatting}")

    df.to_csv(out_filename, sep="\t", index=False)
    print(f"Processed dataset {out_filename}. Total entries: {df.shape[0]}. Invalid formatting: {invalid_formatting}")
    return df