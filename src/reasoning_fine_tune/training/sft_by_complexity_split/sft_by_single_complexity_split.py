import ast
import subprocess
from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.generation.configuration_utils import GenerationConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments

import reasoning_fine_tune.prompts.mmlu_cot_answer as cot_prompts
import reasoning_fine_tune.prompts.mmlu_single_token_answer as prompts
from reasoning_fine_tune.training.sft_by_complexity_split.cot_eval_trainer import CoTEvalTrainer
from reasoning_fine_tune.utils.device import DEVICE_MAP
from reasoning_fine_tune.utils.last_checkpoint_dir import get_last_checkpoint_dir
from reasoning_fine_tune.utils.prepare_dataset import prepare_dataset, prepare_dataset_cot_eval
from reasoning_fine_tune.utils.seed import set_seed

TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 32
LR = 1e-5
EPOCHS = 30


def directory_is_empty(directory: str) -> bool:
    p = Path(directory)
    if not p.exists():
        return True
    if not p.is_dir():
        raise Exception("Not a directory!")
    return not any(p.iterdir())


def get_sys_prompt(row):
    subject = row["base_cluster"]
    return prompts.single_token_sys_prompt(subject)


def get_user_prompt(row):
    question = row["question"]
    options = ast.literal_eval(row["options"])
    return prompts.single_token_answer_prompt(question, options)


def get_sys_prompt_cot_eval(row):
    subject = row["base_cluster"]
    return cot_prompts.cot_sys_prompt(subject)


def get_user_prompt_cot_eval(row):
    question = row["question"]
    options = ast.literal_eval(row["options"])
    return cot_prompts.cot_answer_prompt(question, options)


def train_sft_by_complexity_split(out_path, model_id, train_df_path, test_df_paths, training_kwargs):
    if not directory_is_empty(out_path):
        print("train_sft_by_complexity_split -> out_path not empty", out_path)
        return None

    if training_kwargs is None:
        training_kwargs = {}

    set_seed()

    print(f"Using device: {DEVICE_MAP}")

    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    metrics_accum_correct = 0
    metrics_accum_total = 0
    incorrect_answers: list = []

    def compute_metrics(eval_pred, compute_result, is_cot_eval, question_ids: list[str] | None):
        nonlocal metrics_accum_correct, metrics_accum_total, incorrect_answers

        assert isinstance(question_ids, list)
        assert len(question_ids) != 0

        predictions, labels, inputs = eval_pred.predictions, eval_pred.label_ids, eval_pred.inputs["input_ids"]

        if is_cot_eval:
            # labels are padded from the left
            labels = labels[..., -1:]
            predictions = predictions[:, inputs.shape[1] :]
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            extracted_preds = []
            for p in decoded_preds:
                ans_start = p.find(cot_prompts.answer_marker[0])
                if ans_start != -1:
                    ans_start += len(cot_prompts.answer_marker[0])
                ans_end = p.find(cot_prompts.answer_marker[1])

                if ans_start != -1 and ans_end != -1:
                    extracted_preds.append(p[ans_start:ans_end])
                else:
                    extracted_preds.append("")

            matches = [p.lower() == l.lower() for p, l in zip(extracted_preds, decoded_labels)]

            metrics_accum_correct += sum(matches)
            metrics_accum_total += len(matches)

            for i, is_match in enumerate(matches):
                if is_match:
                    continue

                incorrect_pred = decoded_preds[i]
                incorrect_question_id = question_ids[i]
                incorrect_answers.append(
                    {"is_cot_eval": is_cot_eval, "output": incorrect_pred, "question_id": incorrect_question_id}
                )

        else:
            labels = labels[..., 1:]
            predictions = predictions.argmax(dim=-1)[..., :-1]

            mask = (labels != -100) & (labels != tokenizer.eos_token_id)
            correct = (predictions == labels) & mask

            metrics_accum_correct += correct.sum()
            metrics_accum_total += mask.sum()

            for i, is_match in enumerate(correct):
                if is_match:
                    continue

                incorrect_pred = tokenizer.decode(predictions[i])
                incorrect_question_id = question_ids[i]
                incorrect_answers.append(
                    {"is_cot_eval": is_cot_eval, "output": incorrect_pred, "question_id": incorrect_question_id}
                )

        if not compute_result:
            return None

        accuracy = metrics_accum_correct / metrics_accum_total
        incorrect_answers_res = incorrect_answers
        # Reset for next dataset
        metrics_accum_correct = 0
        metrics_accum_total = 0
        incorrect_answers = []

        return {"accuracy": accuracy, "incorrect_answers": incorrect_answers_res}

    train_df = pd.read_csv(
        train_df_path,
        sep="\t",
        header=0,
    )
    test_dfs = [
        pd.read_csv(
            test_df_path,
            sep="\t",
            header=0,
        )
        for test_df_path in test_df_paths
    ]

    print("Dataframe samples")
    print(train_df.head())
    for test_df in test_dfs:
        print(test_df.head())

    train_ds = prepare_dataset(
        tokenizer=tokenizer, get_sys_prompt=get_sys_prompt, get_user_prompt=get_user_prompt, df=train_df
    )
    # tokenwise eval
    test_tokenwise_ds_dict: dict[str, Dataset] = {
        f"g{i}": prepare_dataset(
            tokenizer=tokenizer,
            get_sys_prompt=get_sys_prompt,
            get_user_prompt=get_user_prompt,
            df=test_df,
            mask_input=True,
        )
        for i, test_df in enumerate(test_dfs)
    }
    test_tokenwise_ds_dict["combined"] = concatenate_datasets(list(test_tokenwise_ds_dict.values()))
    # CoT eval
    test_cot_ds_dict: dict[str, Dataset] = {
        f"g{i}_cot": prepare_dataset_cot_eval(
            tokenizer=tokenizer,
            get_sys_prompt=get_sys_prompt_cot_eval,
            get_user_prompt=get_user_prompt_cot_eval,
            df=test_df,
        )
        for i, test_df in enumerate(test_dfs)
    }
    test_cot_ds_dict["combined_cot"] = concatenate_datasets(list(test_cot_ds_dict.values()))
    # Combined eval dataset
    test_combined_ds_dict = {**test_tokenwise_ds_dict, **test_cot_ds_dict}

    print("Dataset samples")
    print(train_ds[0])
    for test_ds in test_combined_ds_dict.values():
        print(test_ds[0])

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True, pad_to_multiple_of=8, return_tensors="pt"
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=DEVICE_MAP)
    inferred_device_map = model.hf_device_map
    print("\nInferred Device Map:", inferred_device_map)

    generation_config = GenerationConfig.from_pretrained(
        model_id,
        temperature=None,
        top_p=None,
        top_k=None,
        do_sample=False,
        max_new_tokens=1024,
    )
    generation_config.do_sample = False

    training_args = Seq2SeqTrainingArguments(
        seed=42,
        data_seed=42,
        output_dir=out_path,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        bf16=True,
        bf16_full_eval=True,
        logging_strategy="epoch",
        eval_strategy="epoch",
        batch_eval_metrics=True,
        report_to="none",
        save_strategy="epoch",
        overwrite_output_dir=True,
        save_total_limit=1,
        save_only_model=True,
        eval_on_start=True,
        num_train_epochs=EPOCHS,
        lr_scheduler_type="linear",
        learning_rate=LR,
        remove_unused_columns=False,
        include_for_metrics=["inputs"],
        generation_num_beams=1,
        generation_config=generation_config,
        **training_kwargs,
    )
    trainer = CoTEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_combined_ds_dict,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        invalid_answers_save_path=str(Path(out_path).joinpath("incorrect_answers.tsv")),
    )

    trainer.train()

    return get_last_checkpoint_dir(out_path)
