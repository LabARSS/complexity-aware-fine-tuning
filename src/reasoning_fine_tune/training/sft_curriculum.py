import ast
from pathlib import Path

import pandas as pd
import torch
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerState
from transformers.training_args import TrainingArguments

import reasoning_fine_tune.prompts.mmlu_single_token_answer as prompts
from reasoning_fine_tune.utils.prepare_dataset_for_training import prepare_dataset_for_training

BATCH_SIZE = 4

class ArgmaxTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, logits, labels = super().prediction_step(
            model,
            inputs,
            prediction_loss_only=False,      # we need logits once
            ignore_keys=ignore_keys,
        )
        # keep only the class ids; still keep sequence dim if you need one-per-token
        if logits is not None:
            logits = logits.argmax(dim=-1).to(torch.int16)  # tiny!
        return loss, logits, labels


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
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        total = labels.shape[0]

        target_mask = (labels != -100) & (labels != tokenizer.eos_token_id)
        correct_tokens = (predictions == labels) & target_mask

        # We need to account for correct answers only all tokens matched.
        # For instance, 11 could be encoded as two tokens X.
        # Then if model replies 1 (X,), but we expect 11 (X,X), we will count the answer as half-correct, which is wrong
        correct_batches_mask = correct_tokens.sum(-1) == target_mask.sum(-1)
        correct_batches = correct_batches_mask.sum()

        return {"accuracy": correct_batches / total }

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
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

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
        overwrite_output_dir=True,
        save_total_limit=1,
        save_only_model=True,
        eval_on_start=True
    )
    trainer = ArgmaxTrainer(
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
    trainer.state = TrainerState()

    trainer.train()

    training_args.output_dir = str(hard_output_dir)
    trainer.train_dataset = hard_train_ds
    trainer.state = TrainerState()

    trainer.train()

    return trainer.model
