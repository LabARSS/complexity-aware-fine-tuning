import ast
import subprocess
from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers.data.data_collator import DataCollatorForTokenClassification
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

BATCH_SIZE = 2
LR = 1e-5
EPOCHS = 30


def directory_is_empty(directory: str) -> bool:
    p = Path(directory)
    if not p.exists():
        return True
    if not p.is_dir():
        raise Exception("Not a directory!")
    return not any(p.iterdir())


def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)


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

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def compute_metrics(eval_pred, losses):
        if losses is None:
            # CoT eval
            predictions, labels = eval_pred

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
                    extracted_preds.append(None)

            matches = [p == l for p, l in zip(extracted_preds, decoded_labels)]

            return {"accuracy": sum(matches) / len(matches)}
        else:
            # Towenwise eval
            predictions, labels = eval_pred

            labels = labels[..., 1:]
            predictions = predictions[..., :-1]

            mask = (labels != -100) & (labels != tokenizer.eos_token_id)
            correct = (predictions == labels) & mask

            accuracy = correct.sum() / mask.sum()

            return {"accuracy": accuracy}

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

    training_args = Seq2SeqTrainingArguments(
        seed=42,
        data_seed=42,
        output_dir=out_path,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        bf16=True,
        bf16_full_eval=True,
        logging_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
        save_strategy="epoch",
        overwrite_output_dir=True,
        save_total_limit=1,
        save_only_model=True,
        eval_on_start=True,
        num_train_epochs=EPOCHS,
        lr_scheduler_type="linear",
        learning_rate=LR,
        **training_kwargs,
        include_for_metrics=["loss"],
    )
    trainer = CoTEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_combined_ds_dict,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()

    return get_last_checkpoint_dir(out_path)
