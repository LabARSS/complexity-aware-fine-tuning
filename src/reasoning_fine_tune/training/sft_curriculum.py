import ast
import multiprocessing as mp
import subprocess
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

BATCH_SIZE = 4


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


def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)


def get_sys_prompt(row):
    subject = row["base_cluster"]
    return prompts.single_token_sys_prompt(subject)


def get_user_prompt(row):
    question = row["question"]
    options = ast.literal_eval(row["options"])
    return prompts.single_token_answer_prompt(question, options)


def train_sft_curriculum_stage(
    output_subpath, model_id, train_df_path, test_df_path, num_train_epochs, eval_on_start=False
):
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def compute_metrics(eval_pred):
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
    train_df = pd.merge(mmlu_df, train_df["question_id"], on="question_id", how="inner")
    test_df = pd.merge(mmlu_df, test_df["question_id"], on="question_id", how="inner")

    print("Dataframe samples")
    print(train_df.head())
    print(test_df.head())

    train_ds = prepare_dataset(
        tokenizer=tokenizer, get_sys_prompt=get_sys_prompt, get_user_prompt=get_user_prompt, df=train_df
    )
    test_ds = prepare_dataset(
        tokenizer=tokenizer, get_sys_prompt=get_sys_prompt, get_user_prompt=get_user_prompt, df=test_df, mask_input=True
    )

    print("Dataset samples")
    print(train_ds[0])
    print(test_ds[0])

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True, pad_to_multiple_of=8, return_tensors="pt"
    )

    output_dir = Path(__file__).parent.joinpath("../../../artifacts/sft_curriculum").joinpath(output_subpath)

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=DEVICE_MAP)
    inferred_device_map = model.hf_device_map
    print("\nInferred Device Map:", inferred_device_map)

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
        lr_scheduler_type="constant",
        learning_rate=1e-5,
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

    trainer.train()

    return get_last_checkpoint_dir(output_dir)


def _stage_worker(q, kwargs):
    ckpt = train_sft_curriculum_stage(**kwargs)
    q.put(ckpt)


def _run_stage_mp(**kwargs):
    ctx = mp.get_context("spawn")  # safe with CUDA
    q = ctx.Queue()
    p = ctx.Process(target=_stage_worker, args=(q, kwargs))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"stage crashed with {p.exitcode}")
    return Path(q.get())


def train_sft_curriculum(name, model_id, easy_train_df_path, mid_train_df_path, hard_train_df_path, test_df_path):
    np.random.seed(42)
    torch.manual_seed(42)

    print(f"Using device: {DEVICE_MAP}")

    easy_output_dir = _run_stage_mp(
        output_subpath=f"{name}/easy",
        model_id=model_id,
        train_df_path=easy_train_df_path,
        test_df_path=test_df_path,
        num_train_epochs=3,
        eval_on_start=True,
    )

    mid_output_dir = _run_stage_mp(
        output_subpath=f"{name}/mid",
        model_id=easy_output_dir,
        train_df_path=mid_train_df_path,
        test_df_path=test_df_path,
        num_train_epochs=3,
        eval_on_start=False,
    )

    hard_output_dir = _run_stage_mp(
        output_subpath=f"{name}/hard",
        model_id=mid_output_dir,
        train_df_path=hard_train_df_path,
        test_df_path=test_df_path,
        num_train_epochs=4,
        eval_on_start=False,
    )

    return hard_output_dir
