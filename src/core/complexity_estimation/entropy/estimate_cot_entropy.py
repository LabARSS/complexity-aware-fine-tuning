import json

from core.complexity_estimation.entropy.estimate_single_token_entropy import (
    estimate_dataset as estimate_dataset_single_token,
)
from core.complexity_estimation.entropy.logit_sequence_stats import collect_logit_sequence_stats
from core.prompts.mmlu_cot_answer import answer_marker, cot_answer_prompt, cot_sys_prompt
from core.utils.embeddings import get_embeddings

field_ans_token_index = "ans_token_index"
field_entropies_value = "entropies"
field_every_token_info = "every_token_info"
field_input_embeddings = "input_embeddings"
field_think_embeddings = "think_embeddings"
field_answer_embeddings = "answer_embeddings"


def add_cot_entropy_cols(df):
    if field_entropies_value not in df.columns:
        df[field_entropies_value] = ""
    if field_every_token_info not in df.columns:
        df[field_every_token_info] = ""
    if field_ans_token_index not in df.columns:
        df[field_ans_token_index] = -1
    if field_input_embeddings not in df.columns:
        df[field_input_embeddings] = ""
    if field_think_embeddings not in df.columns:
        df[field_think_embeddings] = ""
    if field_answer_embeddings not in df.columns:
        df[field_answer_embeddings] = ""


def process_cot_entropy_response(df, row_idx, inputs, outputs, response_idx, response, model, tokenizer):
    logit_stats = collect_logit_sequence_stats(outputs.scores, response_idx)

    df.at[row_idx, field_entropies_value] = json.dumps(logit_stats.entropies)
    df.at[row_idx, field_every_token_info] = json.dumps(logit_stats.every_token_stats)

    output_str: str = ""
    answer_marker_start = -1
    answer_marker_end = -1
    for i, token in enumerate(logit_stats.greedy_tokens):
        token_str = tokenizer.decode(token)
        output_str += token_str

        if answer_marker_start == -1:
            if answer_marker[0] in output_str:
                answer_marker_start = i
        elif answer_marker_end == -1:
            if answer_marker[1] in output_str:
                answer_marker_end = i

    extracted_answer: str = ""
    if answer_marker_end != -1 and answer_marker_start != -1:
        ans_token_index = answer_marker_start + 1
        extracted_answer = tokenizer.decode(logit_stats.greedy_tokens[ans_token_index])
        df.at[row_idx, field_ans_token_index] = ans_token_index

        answer_embeddings = get_embeddings(model, tokenizer, extracted_answer)
        if answer_embeddings is not None:
            df.at[row_idx, field_answer_embeddings] = json.dumps(answer_embeddings)

        think_text = tokenizer.decode(logit_stats.greedy_tokens[:answer_marker_start])
        think_embeddings = get_embeddings(model, tokenizer, think_text)
        if think_embeddings is not None:
            df.at[row_idx, field_think_embeddings] = json.dumps(think_embeddings)

    formatted_prompt = tokenizer.decode(inputs["input_ids"][response_idx])
    input_embeddings = get_embeddings(model, tokenizer, formatted_prompt)
    if input_embeddings is not None:
        df.at[row_idx, field_input_embeddings] = json.dumps(input_embeddings)

    return extracted_answer


def estimate_dataset(
    in_filename,
    out_filename,
    model,
    tokenizer,
    get_subject_from_row,
    get_question_from_row,
    get_options_from_row,
    check_answer_correct,
    dump_every=1000,
    max_new_tokens=1024,
    get_sys_prompt=cot_sys_prompt,
    get_user_prompt=cot_answer_prompt,
):
    return estimate_dataset_single_token(
        in_filename=in_filename,
        out_filename=out_filename,
        model=model,
        tokenizer=tokenizer,
        get_subject_from_row=get_subject_from_row,
        get_question_from_row=get_question_from_row,
        get_options_from_row=get_options_from_row,
        check_answer_correct=check_answer_correct,
        dump_every=dump_every,
        batch_size=1,
        max_new_tokens=max_new_tokens,
        get_sys_prompt=get_sys_prompt,
        get_user_prompt=get_user_prompt,
        add_columns=add_cot_entropy_cols,
        process_response=process_cot_entropy_response,
    )
