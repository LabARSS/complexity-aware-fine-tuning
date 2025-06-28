from datasets import Dataset


def prepare_dataset_for_training(tokenizer, get_sys_prompt, get_user_prompt, df):
    dataset = Dataset.from_pandas(df)
    return dataset.map(
        lambda row: tokenizer.apply_chat_template(
            [
                {"role": "system", "content": get_sys_prompt(row)},
                {"role": "user", "content": get_user_prompt(row)},
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ),
        remove_columns=["text"],
        num_proc=4,
    )
