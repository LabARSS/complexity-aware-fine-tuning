import torch
from datasets import Dataset


def prepare_dataset_for_training(tokenizer, get_sys_prompt, get_user_prompt, df):
    df["sys_prompt"] = df.apply(get_sys_prompt, axis=1)
    df["user_prompt"] = df.apply(get_user_prompt, axis=1)

    def process_row(row):
        tokenized = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": row["sys_prompt"]},
                {"role": "user", "content": row["user_prompt"]},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        labels = torch.full_like(input_ids, -100)
        labels[-1] = tokenizer.encode(str(row["answer_index"] + 1), add_special_tokens=False)[0]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    dataset = Dataset.from_pandas(df)
    processed_ds = dataset.map(
        process_row,
        num_proc=4,
    )
    processed_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return processed_ds
