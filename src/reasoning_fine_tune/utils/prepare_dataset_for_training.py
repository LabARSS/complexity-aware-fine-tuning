from datasets import Dataset


def prepare_dataset_for_training(tokenizer, get_sys_prompt, get_user_prompt, df):
    df["sys_prompt"] = df.apply(get_sys_prompt, axis=1)
    df["user_prompt"] = df.apply(get_user_prompt, axis=1)

    def process_row(row):
        tokenized_out = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": row["sys_prompt"]},
                {"role": "user", "content": row["user_prompt"]},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        tokenized_out["label"] = tokenizer.encode(str(row["answer_index"] + 1), return_tensors="pt")
        return tokenized_out

    dataset = Dataset.from_pandas(df)
    processed_ds = dataset.map(
        process_row,
        num_proc=4,
    )
    processed_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return processed_ds
