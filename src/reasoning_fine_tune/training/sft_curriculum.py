import ast
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments

import reasoning_fine_tune.prompts.mmlu_single_token_answer as prompts
from reasoning_fine_tune.utils.device import DEVICE_MAP
from reasoning_fine_tune.utils.prepare_dataset import prepare_dataset

BATCH_SIZE = 8


def get_last_checkpoint_dir(path):
    """
    List all direct child directories of *path* and return the one that is
    alphabetically last. Returns None if the directory has no children.

    Examples
    --------
    >>> get_last_checkpoint_dir('/tmp')  # doctest: +SKIP
    PosixPath('/tmp/z_latest')
    """
    p = Path(path)

    if not p.is_dir():
        raise NotADirectoryError(f"{p} is not a directory")

    child_dirs = [d for d in p.iterdir() if d.is_dir()]
    child_dirs.sort()  # alphabetical, case-sensitive

    return child_dirs[-1] if child_dirs else None


def cleaup():
    gc.collect()
    torch.cuda.empty_cache()


def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)


def get_sys_prompt(row):
    subject = row["base_cluster"]
    return prompts.single_token_sys_prompt_with_fallback_for_unknown_answers(subject)


def get_user_prompt(row):
    question = row["question"]
    options = ast.literal_eval(row["options"])
    return prompts.single_token_answer_prompt_with_fallback_for_unknown_answers(question, options)


def train_sft_curriculum(name, model_id, easy_train_df_path, mid_train_df_path, hard_train_df_path, test_df_path):
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"Using device: {DEVICE_MAP}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        labels = labels[..., 1:]
        predictions = predictions[..., :-1]

        mask = (labels != -100) & (labels != tokenizer.eos_token_id)
        correct = (predictions == labels) & mask

        accuracy = correct.sum() / mask.sum()

        return {"accuracy": accuracy}

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

    print("Dataframe samples")
    print(easy_train_df.head())
    print(mid_train_df.head())
    print(hard_train_df.head())
    print(test_df.head())

    print(
        f"Distribution of data (easy:mid:hard:test) = {len(easy_train_df)}:{len(mid_train_df)}:{len(hard_train_df)}:{len(test_df)}"
    )

    easy_train_ds = prepare_dataset(
        tokenizer=tokenizer, get_sys_prompt=get_sys_prompt, get_user_prompt=get_user_prompt, df=easy_train_df
    )
    mid_train_ds = prepare_dataset(
        tokenizer=tokenizer, get_sys_prompt=get_sys_prompt, get_user_prompt=get_user_prompt, df=mid_train_df
    )
    hard_train_ds = prepare_dataset(
        tokenizer=tokenizer, get_sys_prompt=get_sys_prompt, get_user_prompt=get_user_prompt, df=hard_train_df
    )
    test_ds = prepare_dataset(
        tokenizer=tokenizer, get_sys_prompt=get_sys_prompt, get_user_prompt=get_user_prompt, df=test_df, mask_input=True
    )

    print("Dataset samples")
    print(easy_train_ds[0])
    print(test_ds[0])

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True, pad_to_multiple_of=8, return_tensors="pt"
    )

    base_output_dir = Path(__file__).parent.joinpath("../../../artifacts/sft_curriculum").joinpath(name)

    easy_output_dir = base_output_dir.joinpath("easy")
    mid_output_dir = base_output_dir.joinpath("mid")
    hard_output_dir = base_output_dir.joinpath("hard")

    def create_trainer(model, output_dir, train_ds, num_train_epochs, eval_on_start=False):
        training_args = TrainingArguments(
            seed=42,
            data_seed=42,
            output_dir=str(output_dir),
            num_train_epochs=num_train_epochs,
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
            eval_on_start=eval_on_start,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        return trainer

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=DEVICE_MAP)
    inferred_device_map = model.hf_device_map
    print("\nInferred Device Map:", inferred_device_map)

    trainer = create_trainer(
        model=model, output_dir=easy_output_dir, train_ds=easy_train_ds, num_train_epochs=3, eval_on_start=True
    )
    trainer.train()

    # Otherwise, repeated training causes CUDA OOM
    del model
    del trainer
    cleaup()

    model = AutoModelForCausalLM.from_pretrained(get_last_checkpoint_dir(easy_output_dir), device_map=DEVICE_MAP)
    inferred_device_map = model.hf_device_map
    print("\nInferred Device Map:", inferred_device_map)

    trainer = create_trainer(model=model, output_dir=mid_output_dir, train_ds=mid_train_ds, num_train_epochs=3)
    trainer.train()

    del trainer.model
    del trainer
    cleaup()

    model = AutoModelForCausalLM.from_pretrained(get_last_checkpoint_dir(mid_output_dir), device_map=DEVICE_MAP)
    inferred_device_map = model.hf_device_map
    print("\nInferred Device Map:", inferred_device_map)

    trainer = create_trainer(model=model, output_dir=hard_output_dir, train_ds=hard_train_ds, num_train_epochs=4)
    trainer.train()

    del trainer.model
    del trainer
    cleaup()

    return model
