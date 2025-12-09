from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.generation.configuration_utils import GenerationConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments

from peft import PeftModel

import core.prompts.mmlu_cot_answer as cot_prompts
from core.training.sft_by_complexity_split.cot_eval_trainer import CoTEvalTrainer
from core.utils.device import DEVICE_MAP
from core.utils.last_checkpoint_dir import get_last_checkpoint_dir
from core.utils.prepare_dataset import prepare_dataset_cot_eval
from core.utils.seed import set_seed
from core.utils.training_memory_sandbox import memory_sandbox_worker, run_in_memory_sandbox

TRAIN_POSTFIX = "_train.tsv"
TEST_POSTFIX = "_test.tsv"
EVAL_BATCH_SIZE = 4


# Example usage:
#     cot_eval_by_all_complexity_splits(
#     data_folder_path=...,
#     out_path=...,
#     model_id=...,
#     use_lora=True,   
#     checkpoint_steps=[105,112,119,120],   # <- add checkpoint steps to eval here
# )


def _get_sys_prompt_cot_eval(row: pd.Series) -> str:
    subject = row["base_cluster"]
    return cot_prompts.cot_sys_prompt(subject)


def _get_user_prompt_cot_eval(row: pd.Series) -> str:
    question = row["question"]
    options = ast.literal_eval(row["options"])
    return cot_prompts.cot_answer_prompt(question, options)


def _build_cot_eval_datasets(
    tokenizer,
    test_df_paths: list[str | Path],
) -> dict[str, Dataset]:
    test_dfs = [
        pd.read_csv(path, sep="\t", header=0)
        for path in test_df_paths
    ]

    test_cot_ds_dict: dict[str, Dataset] = {
        f"g{i}_cot": prepare_dataset_cot_eval(
            tokenizer=tokenizer,
            get_sys_prompt=_get_sys_prompt_cot_eval,
            get_user_prompt=_get_user_prompt_cot_eval,
            df=test_df,
        )
        for i, test_df in enumerate(test_dfs)
    }

    test_cot_ds_dict["combined_cot"] = concatenate_datasets(
        list(test_cot_ds_dict.values())
    )

    return test_cot_ds_dict


def _load_model_and_tokenizer(
    model_id: str,
    checkpoint_dir: str | Path,
    use_lora: bool | None = None,
):
    checkpoint_dir = Path(checkpoint_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, padding_side="left")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_lora is None:
        use_lora = (checkpoint_dir / "adapter_config.json").exists()

    if use_lora:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=DEVICE_MAP,
        )
        model = PeftModel.from_pretrained(
            base_model,
            checkpoint_dir,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            device_map=DEVICE_MAP,
        )

    print(f"\nUsing device map: {model.hf_device_map}")
    return model, tokenizer


def cot_eval_single_complexity_split(
    *,
    model_id: str,
    checkpoint_dir: str | Path,
    test_df_paths: list[str | Path],
    out_path: str | Path,
    use_lora: bool | None = None,
):
    """
    CoT-validation for one complexity split.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    set_seed()

    model, tokenizer = _load_model_and_tokenizer(
        model_id=model_id,
        checkpoint_dir=checkpoint_dir,
        use_lora=use_lora,
    )

    test_cot_ds_dict = _build_cot_eval_datasets(tokenizer, test_df_paths)

    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    generation_config = GenerationConfig.from_pretrained(
        model_id,
        temperature=None,
        top_p=None,
        top_k=None,
        do_sample=False,
        max_new_tokens=4096,
    )
    generation_config.do_sample = False

    training_args = Seq2SeqTrainingArguments(
        seed=42,
        data_seed=42,
        output_dir=str(out_path),
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        bf16=True,
        bf16_full_eval=True,
        logging_strategy="no",
        eval_strategy="no",
        report_to="none",
        save_strategy="no",
        remove_unused_columns=False,
        include_for_metrics=["inputs"],
        generation_num_beams=1,
        generation_config=generation_config,
        batch_eval_metrics=True,
        do_train=False,
        do_eval=True,
    )

    metrics_accum_correct: float = 0
    metrics_accum_total: float = 0
    incorrect_answers: list[dict[str, Any]] = []

    def compute_metrics(eval_pred, compute_result, is_cot_eval, question_ids: list | None):
        nonlocal metrics_accum_correct, metrics_accum_total, incorrect_answers

        assert isinstance(question_ids, list)
        assert len(question_ids) != 0

        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        inputs = eval_pred.inputs["input_ids"]

        if is_cot_eval:
            labels_cut = labels[..., inputs.shape[1] - 1]
            preds_cut = predictions[:, inputs.shape[1]:]

            decoded_preds = tokenizer.batch_decode(preds_cut, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels_cut, skip_special_tokens=True)

            extracted_preds: list[str] = []
            for p in decoded_preds:
                ans_start = p.find(cot_prompts.answer_marker[0])
                if ans_start != -1:
                    ans_start += len(cot_prompts.answer_marker[0])
                ans_end = p.find(cot_prompts.answer_marker[1])

                if ans_start != -1 and ans_end != -1:
                    extracted_preds.append(p[ans_start:ans_end])
                else:
                    extracted_preds.append("")

            matches = [
                (p or "").strip().lower() == (l or "").strip().lower()
                for p, l in zip(extracted_preds, decoded_labels)
            ]

            metrics_accum_correct += sum(matches)
            metrics_accum_total += len(matches)

            for i, is_match in enumerate(matches):
                if is_match:
                    continue
                incorrect_pred = decoded_preds[i]
                incorrect_question_id = question_ids[i]
                incorrect_answers.append(
                    {
                        "is_cot_eval": True,
                        "output": incorrect_pred,
                        "question_id": incorrect_question_id,
                    }
                )
        else:

            import torch

            labels_cut = labels[..., -1]
            preds_ids = torch.tensor(predictions).argmax(dim=-1)[..., -2]

            correct = preds_ids == labels_cut

            metrics_accum_correct += correct.sum().item()
            metrics_accum_total += len(labels_cut)

            for i, is_match in enumerate(correct):
                if is_match:
                    continue
                incorrect_pred = tokenizer.decode(preds_ids[i])
                incorrect_question_id = question_ids[i]
                incorrect_answers.append(
                    {
                        "is_cot_eval": False,
                        "output": incorrect_pred,
                        "question_id": incorrect_question_id,
                    }
                )
                break

        if not compute_result:
            return None

        accuracy = float(metrics_accum_correct) / float(metrics_accum_total) if metrics_accum_total > 0 else 0.0
        incorrect_answers_res = incorrect_answers

        metrics_accum_correct = 0
        metrics_accum_total = 0
        incorrect_answers = []

        return {"accuracy": accuracy, "incorrect_answers": incorrect_answers_res}

    incorrect_path = out_path / "incorrect_answers_cot.tsv"

    trainer = CoTEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=test_cot_ds_dict,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        invalid_answers_save_path=str(incorrect_path),
    )

    metrics = trainer.evaluate()

    print("\n=== CoT evaluation finished ===")
    print("Checkpoint:", checkpoint_dir)
    print("Metrics:", metrics)
    print("Incorrect answers saved to:", incorrect_path)

    return metrics


@memory_sandbox_worker
def wrapped_cot_eval_single_complexity_split(*args, **kwargs):
    return cot_eval_single_complexity_split(*args, **kwargs)

def cot_eval_by_all_complexity_splits(
    data_folder_path: str,
    out_path: str,
    model_id: str,
    *,
    use_lora: bool | None = None,
    checkpoint_steps: list[int] | None = None,
):
    """
    Cot-eval for all complexity splits found in data_folder_path.
    By default, it takes the last checkpoint of each split.
    If you specify checkpoint_steps=[200, 400], it will evaluate only those steps (if they exist).
    """
    p = Path(data_folder_path)
    if not p.is_dir():
        raise NotADirectoryError(f"{p} is not a directory")

    children = sorted(p.iterdir())  # alphabetical, case-sensitive
    train_df_paths = [c for c in children if c.name.endswith(TRAIN_POSTFIX)]
    test_df_paths = [c for c in children if c.name.endswith(TEST_POSTFIX)]

    if not test_df_paths:
        raise RuntimeError(f"No *{TEST_POSTFIX} files found in {data_folder_path}")

    def _iter_checkpoints_for_split(split_train_out_path: Path) -> Iterable[Path]:
        if checkpoint_steps is None:
            ckpt_dir = get_last_checkpoint_dir(split_train_out_path)
            if ckpt_dir is None:
                print(f"[WARN] No checkpoints found in {split_train_out_path}, skipping this split")
                return []
            return [Path(ckpt_dir)]

        ckpts: list[Path] = []
        for step in checkpoint_steps:
            ckpt_dir = split_train_out_path / f"checkpoint-{step}"
            if ckpt_dir.is_dir():
                ckpts.append(ckpt_dir)
            else:
                print(f"[WARN] {ckpt_dir} not found, skipping")
        return ckpts

    for i, train_df_path in enumerate(train_df_paths):
        split_train_out_path = Path(out_path).joinpath(f"g{i}")

        ckpt_dirs = list(_iter_checkpoints_for_split(split_train_out_path))
        if not ckpt_dirs:
            print(f"[INFO] No checkpoints to eval for split g{i}, skipping")
            continue

        for ckpt_dir in ckpt_dirs:
            step_str = ckpt_dir.name.split("-")[-1] 
            split_eval_out_path = split_train_out_path.joinpath(f"cot_eval_step_{step_str}")

            print(f"=== CoT eval for split g{i}, checkpoint step {step_str} ===")
            print("train file:", train_df_path)
            print("checkpoint:", ckpt_dir)
            print("eval out:", split_eval_out_path)

            run_in_memory_sandbox(
                wrapped_cot_eval_single_complexity_split,
                model_id=model_id,
                checkpoint_dir=str(ckpt_dir),
                test_df_paths=[str(p) for p in test_df_paths],
                out_path=str(split_eval_out_path),
                use_lora=use_lora,
            )