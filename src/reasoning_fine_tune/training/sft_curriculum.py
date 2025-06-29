import ast
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

import reasoning_fine_tune.prompts.mmlu_single_token_answer as prompts
from reasoning_fine_tune.utils.prepare_dataset_for_training import prepare_dataset_for_training

BATCH_SIZE = 4


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

    # Join splits with the original MMLU df as the splits seem to have weird escape chars
    # TODO: Re-do splits!!!
    mmlu_df = pd.read_csv(
        Path(__file__).parent.joinpath("../../../data/source/mmlu_pro_stem.tsv"),
        sep="\t",
        header=0,
    )
    easy_train_df = pd.merge(mmlu_df, easy_train_df["question_id"], on="question_id", how="inner")
    mid_train_df = pd.merge(mmlu_df, mid_train_df["question_id"], on="question_id", how="inner")
    hard_train_df = pd.merge(mmlu_df, hard_train_df["question_id"], on="question_id", how="inner")
    test_df = pd.merge(mmlu_df, test_df["question_id"], on="question_id", how="inner")

    print(easy_train_df.head())
    print(mid_train_df.head())
    print(hard_train_df.head())
    print(test_df.head())
    print(
        f"Distribution of data (easy:mid:hard:test) = {len(easy_train_df)}:{len(mid_train_df)}:{len(hard_train_df)}:{len(test_df)}"
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

    print(easy_train_ds[0])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    base_output_dir = Path(__file__).parent.joinpath("../../../artifacts/sft_curriculum").joinpath(name)

    easy_output_dir = base_output_dir.joinpath("easy")
    mid_output_dir = base_output_dir.joinpath("mid")
    hard_output_dir = base_output_dir.joinpath("hard")

    training_args = TrainingArguments(
        seed=42,
        output_dir=str(easy_output_dir),
        num_train_epochs=5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        bf16=True,
        bf16_full_eval=True,
        logging_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
        save_strategy="epoch",
        lr_scheduler_type="constant",
        overwrite_output_dir=True,
        save_total_limit=1,
        save_only_model=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=easy_train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    training_args.output_dir = str(mid_output_dir)
    trainer.train_dataset = mid_train_ds

    trainer.train()

    training_args.output_dir = str(hard_output_dir)
    trainer.train_dataset = hard_train_ds

    trainer.train()

    return trainer.model
