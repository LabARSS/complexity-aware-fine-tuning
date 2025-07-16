from reasoning_fine_tune.prompts.mmlu_option_ids import option_ids


def check_answer_correct_mmlu(row, model_answer):
    try:
        return option_ids[int(row["answer_index"])] == model_answer.strip()
    except:
        return False
