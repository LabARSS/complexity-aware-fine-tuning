from datasets import Dataset


def prepare_dataset_for_training(tokenizer, get_sys_prompt, get_user_prompt, df):
    df["sys_prompt"] = df.apply(get_sys_prompt, axis=1)
    df["user_prompt"] = df.apply(get_user_prompt, axis=1)

    def process_row(row):
        input_ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": row["sys_prompt"]},
                {"role": "user", "content": row["user_prompt"]},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )[0]
        label = tokenizer.encode(str(row["answer_index"] + 1), return_tensors="pt")[0]
        return {
            'input_ids': input_ids,
            'label': label
        }

    dataset = Dataset.from_pandas(df)
    processed_ds = dataset.map(
        process_row,
        num_proc=4,
    )
    processed_ds.set_format(type="torch", columns=["input_ids", "label"])

    return processed_ds
