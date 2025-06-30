from datasets import Dataset


def prepare_dataset(tokenizer, get_sys_prompt, get_user_prompt, df, mask_input=False):
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
            return_dict=True,
        )

        answer_id = tokenizer.encode(str(row["answer_index"] + 1), add_special_tokens=False)[0]

        input_ids = tokenized["input_ids"] + [answer_id]
        attention_mask = tokenized["attention_mask"] + [1]

        labels = input_ids.copy()

        if mask_input:
            for i in range(len(tokenized["input_ids"])):
                labels[i] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    dataset = Dataset.from_pandas(df)

    processed_ds = dataset.map(
        process_row,
        num_proc=4,
        remove_columns=dataset.column_names,
    )

    return processed_ds
