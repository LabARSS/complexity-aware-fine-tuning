import ast
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

import reasoning_fine_tune.prompts.mmlu_single_token_answer as prompts
from reasoning_fine_tune.utils.prepare_dataset_for_training import prepare_dataset_for_training


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.argmax(predictions, dim=-1)
    acc, prec, rec, f1 = (
        accuracy_score(labels, predictions),
        *precision_recall_fscore_support(labels, predictions, average="binary")[:3],
    )
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


def get_sys_prompt(row):
    subject = row["base_cluster"]
    return prompts.single_token_sys_prompt_with_fallback_for_unknown_answers(subject)


def get_user_prompt(row):
    question = row["question"]
    options = ast.literal_eval(row["options"])
    return prompts.single_token_answer_prompt_with_fallback_for_unknown_answers(question, options)


def train_sft_curriculum(
    name, model, tokenizer, easy_train_df_path, mid_train_df_path, hard_train_df_path, test_df_path
):
    easy_train_df = pd.read_csv(
        easy_train_df_path,
        sep="\t",
        header=0,
    )
    mid_train_df = pd.read_csv(
        mid_train_df_path,
        sep="\t",
        header=0,
    )
    hard_train_df = pd.read_csv(
        hard_train_df_path,
        sep="\t",
        header=0,
    )
    test_df = pd.read_csv(
        test_df_path,
        sep="\t",
        header=0,
    )

    easy_train_ds = prepare_dataset_for_training(
        tokenizer=tokenizer, get_sys_prompt=get_sys_prompt, get_user_prompt=get_user_prompt, df=easy_train_df
    )
    mid_train_ds = prepare_dataset_for_training(
        tokenizer=tokenizer, get_sys_prompt=get_sys_prompt, get_user_prompt=get_user_prompt, df=mid_train_df
    )
    hard_train_ds = prepare_dataset_for_training(
        tokenizer=tokenizer, get_sys_prompt=get_sys_prompt, get_user_prompt=get_user_prompt, df=hard_train_df
    )
    test_ds = prepare_dataset_for_training(
        tokenizer=tokenizer, get_sys_prompt=get_sys_prompt, get_user_prompt=get_user_prompt, df=test_df
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    base_output_dir = Path(__file__).parent.joinpath("../../../artifacts/sft_curriculum").joinpath(name)

    easy_output_dir = base_output_dir.joinpath("easy")
    mid_output_dir = base_output_dir.joinpath("mid")
    hard_output_dir = base_output_dir.joinpath("hard")

    training_args_easy = TrainingArguments(
        seed=42,
        output_dir=str(easy_output_dir),
        num_train_epochs=5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        bf16=True,
        bf16_full_eval=True,
        logging_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
        save_strategy="epoch",
        overwrite_output_dir=True,
        save_total_limit=1,
        save_only_model=True,
    )
    trainer_easy = Trainer(
        model=model,
        args=training_args_easy,
        train_dataset=easy_train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer_easy.train()

    training_args_mid = training_args_easy.update({"output_dir": mid_output_dir})
    trainer_mid = Trainer(
        model=trainer_easy.model,
        args=training_args_mid,
        train_dataset=mid_train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer_mid.train()

    training_args_hard = training_args_easy.update({"output_dir": hard_output_dir})
    trainer_hard = Trainer(
        model=trainer_mid.model,
        args=training_args_hard,
        train_dataset=hard_train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer_hard.train()

    return trainer_hard.model
