from reasoning_fine_tune.prompts.mmlu_option_ids import option_ids, option_ids_w_fallback


def validate_mmlu_answer(answer: str | int):
    return str(answer) in option_ids_w_fallback


def keep_only_valid_and_known_answers(df, column_name, option_ids=option_ids):
    return df[df[column_name].isin(option_ids)]
