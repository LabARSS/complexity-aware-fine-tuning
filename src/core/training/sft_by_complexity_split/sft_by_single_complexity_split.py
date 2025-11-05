import ast
import json
import subprocess
from pathlib import Path

import pandas as pd
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig, TaskType, get_peft_model
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.generation.configuration_utils import GenerationConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments

import core.prompts.mmlu_cot_answer as cot_prompts
import core.prompts.mmlu_single_token_answer as prompts
from core.training.sft_by_complexity_split.cot_eval_trainer import CoTEvalTrainer
from core.training.callbacks.save_and_log_weights import SaveOnEpochEndAndLogWeightsCallback
from core.utils.device import DEVICE_MAP
from core.utils.last_checkpoint_dir import get_last_checkpoint_dir
from core.utils.prepare_dataset import prepare_dataset, prepare_dataset_cot_eval
from core.utils.seed import set_seed

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
LR = 1e-5
EPOCHS = 20


def directory_is_empty(directory: str, expected_epochs: int) -> bool:
    p = Path(directory)
    if not p.exists():
        return True
    if not p.is_dir():
        raise Exception("Not a directory!")

    checkpoint_dirs = list(p.glob("checkpoint-*"))
    if not checkpoint_dirs:
        return True

    checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[1]))
    last_checkpoint = checkpoint_dirs[-1] if checkpoint_dirs else None

    if last_checkpoint:
        state_file = last_checkpoint / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)
                if int(state.get("epoch", 0)) == expected_epochs:
                    return False

    return True


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


# helper for default LoRA
def _build_lora_config(model, lora_kwargs: dict | None) -> LoraConfig:
    lora_kwargs = lora_kwargs or {}
    default_target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    target_modules = lora_kwargs.get("target_modules", default_target_modules)

    return LoraConfig(
        r=lora_kwargs.get("r", 16),
        lora_alpha=lora_kwargs.get("lora_alpha", 32),
        lora_dropout=lora_kwargs.get("lora_dropout", 0.05),
        bias=lora_kwargs.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        # Additionaly can be used:
        # modules_to_save=lora_kwargs.get("modules_to_save"),
        use_rslora=lora_kwargs.get("use_rslora", True),
        # init_lora_weights=lora_kwargs.get("init_lora_weights", True),
    )


def train_sft_by_complexity_split(
    out_path,
    model_id,
    train_df_path,
    test_df_paths,
    training_kwargs,
    *,
    use_lora: bool = False,
    lora_kwargs: dict | None = None,  # LoRA params
):
    if not directory_is_empty(out_path, EPOCHS):
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

    def compute_metrics(eval_pred, compute_result, is_cot_eval, question_ids: list | None):
        nonlocal metrics_accum_correct, metrics_accum_total, incorrect_answers

        assert isinstance(question_ids, list)
        assert len(question_ids) != 0

        predictions, labels, inputs = eval_pred.predictions, eval_pred.label_ids, eval_pred.inputs["input_ids"]

        if is_cot_eval:
            # labels are padded from both left and right (weird, I know)
            labels = labels[..., inputs.shape[1] - 1]
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
            labels = labels[..., -1]
            predictions = predictions.argmax(dim=-1)[..., -2]

            correct = predictions == labels

            metrics_accum_correct += correct.sum()
            metrics_accum_total += len(labels)

            for i, is_match in enumerate(correct):
                if is_match:
                    continue

                incorrect_pred = tokenizer.decode(predictions[i])
                incorrect_question_id = question_ids[i]
                incorrect_answers.append(
                    {"is_cot_eval": is_cot_eval, "output": incorrect_pred, "question_id": incorrect_question_id}
                )
                break

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

    if use_lora:
        lora_config = _build_lora_config(model, lora_kwargs)
        model = get_peft_model(model, lora_config)
        # Удобный лог о числе обучаемых параметров
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

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
        save_only_model=True,  # with peft save only adapters
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

    trainer.add_callback(
        SaveOnEpochEndAndLogWeightsCallback(
            output_dir=out_path,
            save_full_model_for_non_lora=False,
        )
    )

    trainer.train()

    return get_last_checkpoint_dir(out_path)
