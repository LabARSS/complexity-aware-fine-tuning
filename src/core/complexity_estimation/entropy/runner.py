import gc
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from core.complexity_estimation.entropy.logit_entropy import compute_entropy_from_logits
from core.complexity_estimation.entropy.logit_sequence_stats import collect_logit_sequence_stats
from core.utils.device import DEVICE, move_batch_to_device
from core.utils.validation import validate_mmlu_answer as default_validate_mmlu_answer
from core.utils.embeddings import get_embeddings

Mode = Literal["single_token", "cot"]
EntropyReducer = Literal["last_step", "sequence"]

@dataclass
class EntropyConfig:
    mode: Mode
    max_new_tokens: int
    entropy_reducer: EntropyReducer
    prompt_builder: Callable[[pd.Series], List[Dict[str, str]]]
    answer_extractor: Callable[[List[int], str, PreTrainedTokenizerBase], Dict[str, Optional[str]]]
    check_answer_correct: Callable[[pd.Series, str], bool]
    validate_answer: Callable[[str], bool] = default_validate_mmlu_answer
    batch_size: int = 16
    dump_every: int = 1000
    compute_input_embeddings: bool = False
    compute_think_answer_embeddings: bool = False
    override_columns: Optional[Dict[str, str]] = None  # name -> column_name

def _default_columns(model_name: str, mode: Mode) -> Dict[str, str]:
    # единый шаблон
    cols = {
        "ans": f"{model_name}_ans",
        "ans_correct": f"{model_name}_ans_correct",
        "entropy": f"{model_name}_entropy", #single-token
        "entropies": f"{model_name}_entropies", # CoT
        "every_token_info": f"{model_name}_every_token_info",
        "response": f"{model_name}_response",
        "ans_token_index": f"{model_name}_ans_token_index",
        "input_embeddings": f"{model_name}_input_embeddings",
        "think_embeddings": f"{model_name}_think_embeddings",
        "answer_embeddings": f"{model_name}_answer_embeddings",
    }
    if mode == "single_token":
        for k in ["entropies", "every_token_info", "response", "ans_token_index",
                  "think_embeddings", "answer_embeddings"]:
            cols.pop(k, None)
    return cols

def _ensure_columns(df: pd.DataFrame, cols: Dict[str, str], mode: Mode) -> None:
    if cols.get("ans") and cols["ans"] not in df.columns:
        df[cols["ans"]] = ""
    if cols.get("ans_correct") and cols["ans_correct"] not in df.columns:
        df[cols["ans_correct"]] = False
    if mode == "single_token":
        if cols.get("entropy") and cols["entropy"] not in df.columns:
            df[cols["entropy"]] = 0.0
    else:
        for name in ["entropies", "every_token_info", "response", "ans_token_index",
                     "input_embeddings", "think_embeddings", "answer_embeddings"]:
            col = cols.get(name)
            if col and col not in df.columns:
                if name == "ans_token_index":
                    df[col] = -1
                else:
                    df[col] = ""

def _save_df(df: pd.DataFrame, out_filename: str) -> None:
    if out_filename.lower().endswith(".parquet"):
        df.to_parquet(out_filename, compression="gzip")
    elif out_filename.lower().endswith(".tsv"):
        df.to_csv(out_filename, sep="\t", index=False)
    else:
        df.to_csv(out_filename, index=False)

def run_entropy_estimation(
    in_filename: str,
    out_filename: str,
    model,
    tokenizer: PreTrainedTokenizerBase,
    cfg: EntropyConfig,
    get_subject_from_row: Callable[[pd.Series], str],
    get_question_from_row: Callable[[pd.Series], str],
    get_options_from_row: Callable[[pd.Series], str],
    resume: bool = True,
    num_proc: int = 4,
) -> pd.DataFrame:
    if resume and os.path.exists(out_filename):
        df = pd.read_parquet(out_filename) if out_filename.endswith(".parquet") else pd.read_csv(out_filename, sep="\t")
    else:
        df = pd.read_csv(in_filename, sep="\t", header=0)

    model_name = model.config_class().model_type
    print(model_name)
    cols = _default_columns(model_name, cfg.mode)
    if cfg.override_columns:
        cols.update(cfg.override_columns)
    _ensure_columns(df, cols, cfg.mode)


    base_df = df.reset_index().rename(columns={"index": "orig_index"})
    def preprocess_row(row: pd.Series) -> Dict[str, torch.Tensor]:
        messages = cfg.prompt_builder(row)
        tokenized = tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        out = {k: v.squeeze(0) for k, v in tokenized.items()}
        out["orig_index"] = torch.tensor(int(row["orig_index"]))
        return out

    if cfg.mode == "cot" and cols.get("ans_token_index") in df.columns:
        mask = df[cols["ans_token_index"]] == -1
        work_df = base_df[mask.reset_index(drop=True)]
    else:
        work_df = base_df

    if len(work_df) == 0:
        print("len(work_df) == 0")
        return df

    ds = Dataset.from_pandas(work_df)
    ds = ds.map(
        preprocess_row, num_proc=num_proc, batched=False,
        remove_columns=[c for c in ds.column_names if c not in ["input_ids", "attention_mask", "orig_index"]],
    )

    # батчи:
    data_collator = DataCollatorWithPadding(tokenizer)
    dataloader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=data_collator)

    tokenizer.padding_side = "left"
    pbar = tqdm(dataloader)

    correct = 0
    invalid = 0
    seen = 0

    for batch_idx, batch in enumerate(pbar):
        gc.collect()
        if DEVICE == torch.device("cuda"):
            torch.cuda.empty_cache()

        batch = move_batch_to_device(batch, DEVICE)
        outputs = model.generate(
            **batch,
            max_new_tokens=cfg.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            temperature=None,
            top_p=None,
            top_k=None,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
        input_len = batch["input_ids"].shape[1]
        gen_token_batch = outputs.sequences[:, input_len:]  # [B, T_new]
        decoded_batch = tokenizer.batch_decode(gen_token_batch, skip_special_tokens=True)

        B = gen_token_batch.shape[0]
        for i in range(B):
            orig_idx = int(batch["orig_index"][i].item())
            gen_ids: List[int] = gen_token_batch[i].tolist()
            gen_text: str = decoded_batch[i]

            # Энтропия:
            if cfg.entropy_reducer == "last_step":
                last_logits = outputs.scores[-1][i]
                entropy_value = float(compute_entropy_from_logits(last_logits))
                df.at[orig_idx, cols["entropy"]] = entropy_value
            elif cfg.entropy_reducer == "sequence":
                # Собираем логиты для конкретного примера i
                per_step = tuple(step_logits[i] for step_logits in outputs.scores) 
                # per_step = tuple(step_logits[i].unsqueeze(0) for step_logits in outputs.scores)
                stats = collect_logit_sequence_stats(per_step)
                df.at[orig_idx, cols["entropies"]] = json.dumps(stats.entropies)
                df.at[orig_idx, cols["every_token_info"]] = json.dumps(stats.every_token_stats)
                if "response" in cols:
                    df.at[orig_idx, cols["response"]] = gen_text
            else:
                raise NotImplementedError(f"Expected 'last_step' or 'sequence'. Get: {cfg.entropy_reducer}")

            ans_info = cfg.answer_extractor(gen_ids, gen_text, tokenizer) or {}
            answer = (ans_info.get("answer") or "").strip()
            if cols.get("ans"):
                df.at[orig_idx, cols["ans"]] = answer
            if "ans_token_index" in cols and ans_info.get("ans_token_index") is not None:
                df.at[orig_idx, cols["ans_token_index"]] = int(ans_info["ans_token_index"])

            # Эмбеддинги
            if cfg.compute_input_embeddings:
                messages = cfg.prompt_builder(df.iloc[orig_idx])
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inp_emb = get_embeddings(model, tokenizer, prompt_text)
                if inp_emb is not None and "input_embeddings" in cols:
                    df.at[orig_idx, cols["input_embeddings"]] = json.dumps(inp_emb)

            if cfg.mode == "cot" and cfg.compute_think_answer_embeddings:
                think_text = ans_info.get("think_text")
                if think_text:
                    think_emb = get_embeddings(model, tokenizer, think_text)
                    if think_emb is not None and "think_embeddings" in cols:
                        df.at[orig_idx, cols["think_embeddings"]] = json.dumps(think_emb)
                if answer:
                    ans_emb = get_embeddings(model, tokenizer, answer)
                    if ans_emb is not None and "answer_embeddings" in cols:
                        df.at[orig_idx, cols["answer_embeddings"]] = json.dumps(ans_emb)

            # Валидация
            if cfg.validate_answer(answer):
                ok = cfg.check_answer_correct(df.iloc[orig_idx], answer)
                df.at[orig_idx, cols["ans_correct"]] = bool(ok)
                if ok:
                    correct += 1
            else:
                invalid += 1

            seen += 1

        # прогрессбар
        if seen:
            pbar.set_description(f"accuracy={correct/seen:.2f} / invalid={invalid}")

        # периодические дампы
        if cfg.dump_every and (seen % cfg.dump_every == 0):
            _save_df(df, out_filename)

    _save_df(df, out_filename)
    print(f"Processed {out_filename}. Total entries: {df.shape[0]}. Invalid: {invalid}")
    return df
