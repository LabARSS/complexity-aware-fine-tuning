import gc
import os
from typing import Callable, cast

import pandas as pd
import torch
from pydantic.config import ConfigDict
from pydraconf.base_config import PydraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning_fine_tune.entropy_estimation.logit_entropy import compute_entropy_from_logits
from reasoning_fine_tune.prompts.gsm8k_cot_answer import answer_marker


class EstimateDatasetConfig(PydraConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    in_filename: str
    out_filename: str
    dump_every: int = 100
    max_new_tokens: int = 1
    get_sys_prompt: Callable[[pd.Series], str]
    get_user_prompt: Callable[[pd.Series], str]
    model_name: str
    device: str
    model_config_dict: dict
    check_answer_correct: Callable[[pd.Series, str], bool]


def estimate_dataset(config: EstimateDatasetConfig):
    invalid_answers = 0

    in_filename = config.in_filename
    file_type = os.path.splitext(config.in_filename)[1]

    if os.path.exists(config.out_filename):
        in_filename = config.out_filename

    if file_type == ".jsonl":
        df = pd.read_json(in_filename, lines=True)
    else:
        df = pd.read_csv(
            in_filename,
            sep="\t",
            header=0,
        )

    field_ans = "entropy_ans"
    field_ans_correct = "entropy_ans_correct"
    field_entropy_value = "entropy_value"

    if field_ans_correct not in df.columns:
        df[field_ans_correct] = False
    if field_entropy_value not in df.columns:
        df[field_entropy_value] = 0.0
    if field_ans not in df.columns:
        df[field_ans] = ""

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **config.model_config_dict).to(config.device)

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row[field_ans] != "":
            continue

        gc.collect()
        if config.device == "cuda":
            torch.cuda.empty_cache()

        # print(f"loop {index} -> start: {model.get_memory_footprint(return_buffers=True) / 10**9} GB")

        sys_prompt = config.get_sys_prompt(row)
        user_prompt = config.get_user_prompt(row)

        # print(user_prompt)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(config.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            temperature=None,
            top_p=None,
            top_k=None,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        # print(f"loop {index} -> after generate: {model.get_memory_footprint(return_buffers=True) / 10**9} GB")

        input_length = inputs.input_ids.shape[1]
        answer_raw = outputs.sequences[0, input_length:]

        answer_token_map = []
        answer = ""
        for i, token in enumerate(answer_raw):
            token_decoded = tokenizer.decode(token.unsqueeze(0), skip_special_tokens=True)
            answer_token_map.extend([i] * len(token_decoded))
            answer += token_decoded

        df.at[index, field_ans] = answer

        answer_marker_start = answer.find(answer_marker[0])
        answer_marker_end = answer.find(answer_marker[1])

        extracted_answer_position = -1
        extracted_answer = ""
        if answer_marker_end != -1 and answer_marker_start != -1:
            extracted_answer = answer[answer_marker_start + len(answer_marker[0]) : answer_marker_end]
            extracted_answer_position = answer_marker_start + len(answer_marker[0])

        if extracted_answer_position == -1:
            invalid_answers += 1
            continue

        extracted_answer_position = answer_token_map[extracted_answer_position]

        # generated token position, batch_dim
        final_token_logits = outputs.scores[extracted_answer_position][0]
        entropy = compute_entropy_from_logits(final_token_logits)
        df.at[index, field_entropy_value] = entropy

        try:
            df.at[index, field_ans] = extracted_answer
            df.at[index, field_ans_correct] = config.check_answer_correct(df.iloc[index], extracted_answer)
        except Exception:
            invalid_answers += 1

        if index < 5:
            print(
                f"Answer: {answer}\nExtracted answer: {extracted_answer}\nAnswer position: {extracted_answer_position}/{len(outputs.scores)}\nExtracted answer token: {answer_raw[extracted_answer_position]} ({tokenizer.decode(answer_raw[extracted_answer_position].unsqueeze(0))})\nEntropy: {df.at[index, field_entropy_value]}\nis_correct: {df.at[index, field_ans_correct]}\n\n\n"
            )

        if cast(int, index) % config.dump_every == 0:
            if file_type == ".jsonl":
                df.to_json(config.out_filename, lines=True, orient="records")
            else:
                df.to_csv(config.out_filename, sep="\t", index=False)

            print(
                f"Processing dataset {config.out_filename}... Processed: {index}/{df.shape[0]}. Invalid answers: {invalid_answers}"
            )

    if file_type == ".jsonl":
        df.to_json(config.out_filename, lines=True, orient="records")
    else:
        df.to_csv(config.out_filename, sep="\t", index=False)
    print(f"Processed dataset {config.out_filename}. Total entries: {df.shape[0]}. Invalid answers: {invalid_answers}")
    return df
