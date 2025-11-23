import json
import logging
import os
import sys
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from reasoning_fine_tune.training.sft.config import TrainConfig
from reasoning_fine_tune.training.sft.utils import ANSWER_MARKER, ANSWER_PATTERN, calculate_accuracy, cot_sys_prompt


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self._setup_logging()
        print("Trainer init: device check...", flush=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        self.model = None
        self.tokenizer = None

    def _setup_logging(self):
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        log_file = os.path.join(self.cfg.save_dir, f"training_{os.getpid()}.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler(sys.stdout)],
            force=True,
        )
        print(f"Logging configured. log_file={log_file}", flush=True)

        self.results_file = os.path.join(self.cfg.save_dir, f"training_{os.getpid()}_results.jsonl")
        if os.path.exists(self.results_file):
            try:
                os.remove(self.results_file)
                logging.debug(f"Removed previous results file: {self.results_file}")
            except Exception:
                pass

    def _add_result(self, desc: str, epoch: int, accuracy: float):
        with open(self.results_file, "a") as f:
            result_dict = {"desc": desc, "epoch": epoch, "accuracy": accuracy}
            f.write(json.dumps(result_dict) + "\n")

    def load_model_and_tokenizer(self):
        # debug prints so we see progress in console
        print(f"Loading model {self.cfg.base_model} (may take long)...", flush=True)
        logging.info(f"Loading model {self.cfg.base_model}...")
        device_map = getattr(self.cfg, "device_map", "auto")
        if device_map == "auto" and not torch.cuda.is_available():
            logging.info("CUDA not available, switching device_map to None for safe load.")
            device_map = None

        # fallback dtype if bfloat16 is not supported on this environment
        torch_dtype = getattr(torch, "bfloat16", None) or torch.float32

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.base_model,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                use_cache=False,
            )
            print("Model.from_pretrained returned successfully.", flush=True)
            logging.info("Model loaded from_pretrained")
        except Exception as e:
            print("Exception during from_pretrained:", repr(e), flush=True)
            logging.exception("Exception in from_pretrained")
            raise
            # Try to enable gradient checkpointing if possible

        self.model.gradient_checkpointing_enable()

        # LoRA wrapping
        if getattr(self.cfg, "use_lora", False):
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError:
                logging.error("use_lora=True, но пакет `peft` не установлен. Установи `pip install peft`.")
                raise

            lora_config = LoraConfig(
                r=self.cfg.lora_r,
                lora_alpha=self.cfg.lora_alpha,
                lora_dropout=self.cfg.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=list(self.cfg.lora_target_modules),
            )
            self.model = get_peft_model(self.model, lora_config)
            logging.info("Model wrapped with LoRA.")
            self.model.print_trainable_parameters()

            # If device_map was None (no automatic placement), move model to device
            if device_map is None:
                try:
                    self.model.to(self.device)
                    logging.info(f"Moved model to device {self.device}")
                except Exception as e:
                    logging.warning(f"Failed to move model to device: {e}")

            # tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model, use_fast=True)
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            print("Tokenizer loaded.", flush=True)
            logging.info("Tokenizer loaded")

    # prompt formatting
    def format_prompt_qa(self, row: pd.Series, include_answer: bool = True, new_format: bool = True) -> str:
        if new_format:
            try:
                if "question" not in row.index:
                    return None

                question = str(row["question"]).strip()
                if not question:
                    return None

                sys_msg = cot_sys_prompt(self.cfg.use_cot)
                t = self.cfg.prompt_tokens

                # system + user часть
                prompt = (
                    f"{t['system_start']}\n {sys_msg}{t['system_end']} \n{t['user_start']}\n Question: {question} \n"
                )

                if self.cfg.use_cot:
                    prompt += "Please analyze step by step and provide the answer."
                else:
                    prompt += (
                        "Analyze question and answer with only one number in "
                        f"{ANSWER_MARKER[0]}number{ANSWER_MARKER[1]} format."
                    )

                prompt += t["user_end"] + "\n" + t["assistant_start"]

                if not include_answer:
                    return prompt

                answer = None
                if "answer" in row.index and pd.notna(row["answer"]):
                    answer = str(row["answer"]).strip()
                    if answer == "":
                        answer = None

                if self.cfg.use_cot:
                    cot_text = None
                    if "distill_response" in row.index and pd.notna(row["distill_response"]):
                        cot_text = str(row["distill_response"]).strip()
                    elif "reasoning" in row.index and pd.notna(row["reasoning"]):
                        cot_text = str(row["reasoning"]).strip()

                    if answer:
                        prompt += f"\n {cot_text} \n {ANSWER_MARKER[0]}{answer}{ANSWER_MARKER[1]}{t['assistant_end']}"
                    else:
                        prompt += f"\n {cot_text}{t['assistant_end']}"
                        
                elif answer:
                    prompt += f"\n {ANSWER_MARKER[0]}{answer}{ANSWER_MARKER[1]}{t['assistant_end']}"
                else:
                    prompt += t["assistant_end"]

                return prompt

            except Exception as e:
                logging.debug(f"Failed to format prompt for row: {e}")
                return None

        else:
            # old code for tsv
            try:
                options = literal_eval(row["options"])
                # display: "1. option0", "2. option1", ...
                formatted_options = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])
                N = len(options)

                sys_msg = cot_sys_prompt(self.cfg.use_cot)
                t = self.cfg.prompt_tokens
                prompt = (
                    f"{t['system_start']}\n {sys_msg}{t['system_end']} \n"
                    f"{t['user_start']}\n Question: {row['question']} \nOptions: \n{formatted_options} \n"
                )
                if self.cfg.use_cot:
                    prompt += "Please analyze step by step and provide the answer."
                else:
                    prompt += (
                        "Please analyze and provide the answer with only one number in "
                        f"{ANSWER_MARKER[0]}number{ANSWER_MARKER[1]}."
                    )

                prompt += t["user_end"] + "\n" + t["assistant_start"]

                if include_answer:
                    if self.cfg.use_cot and "distill_response" in row.index and pd.notna(row["distill_response"]):
                        prompt += f"\n{row['distill_response']}{t['assistant_end']}"
                    elif "answer_index" in row.index and pd.notna(row["answer_index"]):
                        # answer_index is expected to be zero-based already
                        prompt += (
                            f"\n{ANSWER_MARKER[0]}{int(row['answer_index']) + 1}{ANSWER_MARKER[1]}{t['assistant_end']}"
                        )
                    else:
                        prompt += t["assistant_end"]
                return prompt
            except Exception as e:
                logging.debug(f"Failed to format prompt for row: {e}")
                return None

    # data preparation
    def prepare_datasets(self, train_file: str | list[str], val_file: str, test_file: str):
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        def load_and_filter_df(
            path: str | list[str],
            require_distill: bool = False,
            sample_size: int | None = None,
        ):
            def _load_one(p: str) -> pd.DataFrame:
                logging.info(f"Loading dataset: {p}")
                if p.endswith(".jsonl") or p.endswith(".json"):
                    return pd.read_json(p, lines=True)
                else:
                    # old tsv format still
                    return pd.read_csv(p, sep="\t")

            if isinstance(path, str):
                df = _load_one(path)
            else:
                dfs = []
                for p in path:
                    dfs.append(_load_one(p))
                df = pd.concat(dfs, ignore_index=True)

            total_rows = len(df)

            if require_distill:
                if "distill_response" in df.columns:
                    df = df[df["distill_response"].notna()]
                elif "reasoning" in df.columns:
                    df = df[df["reasoning"].notna()]

            formatted = df.apply(lambda r: self.format_prompt_qa(r, include_answer=True), axis=1)
            valid_mask = formatted.notnull()

            df = df[valid_mask].reset_index(drop=True)
            formatted = formatted[valid_mask].reset_index(drop=True)

            logging.info(
                f"Skipped {total_rows - valid_mask.sum()} rows when preparing {path} (after CoT/filtering: {len(df)})"
            )

            # Sample if needed
            if sample_size is not None and len(df) > sample_size:
                idx = np.random.choice(len(df), size=sample_size, replace=False)
                df = df.iloc[idx].reset_index(drop=True)
                formatted = formatted.iloc[idx].reset_index(drop=True)
                logging.info(f"Using sample of {sample_size} rows from {path} (total after filter: {len(valid_mask)})")

            return df, formatted

        train_sample_size = getattr(self.cfg, "train_sample_size", None)
        val_sample_size = getattr(self.cfg, "val_sample_size", None)
        test_sample_size = getattr(self.cfg, "test_sample_size", None)

        train_df, train_formatted = load_and_filter_df(
            train_file,
            require_distill=self.cfg.use_cot,
            sample_size=train_sample_size,
        )
        val_df, val_formatted = load_and_filter_df(
            val_file,
            require_distill=self.cfg.use_cot,
            sample_size=val_sample_size,
        )
        test_df, test_formatted = load_and_filter_df(
            test_file,
            require_distill=self.cfg.use_cot,
            sample_size=test_sample_size,
        )

        # top-level, module-level function (pickleable)
        def tokenize_batch(examples, tokenizer_name: str, max_length: int):
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            if getattr(tok, "pad_token", None) is None:
                tok.pad_token = tok.eos_token
            return tok(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=True,
                add_special_tokens=False,
            )

        def create_dataset(formatted_series, _orig_df):
            dataset = Dataset.from_pandas(pd.DataFrame({"text": formatted_series}), preserve_index=False)
            return dataset.map(
                tokenize_batch,
                batched=True,
                batch_size=512,
                remove_columns=["text"],
                num_proc=min(4, self.cfg.num_workers),
                fn_kwargs={"tokenizer_name": self.cfg.base_model, "max_length": self.cfg.max_input_length},
            ), _orig_df

        train_ds, train_df = create_dataset(train_formatted, train_df)
        val_ds, val_df = create_dataset(val_formatted, val_df)
        test_ds, test_df = create_dataset(test_formatted, test_df)

        return train_ds, val_ds, test_ds, train_df, val_df, test_df

    # evaluatio
    def evaluate_qa(
        self,
        model,
        df: pd.DataFrame,
        tokenizer,
        epoch: int,
        desc: str = "Validating",
        data_format: str = "jsonl",
    ) -> float:
        model.eval()
        total_correct = 0
        total_errors = 0
        processed = 0
        pbar = tqdm(total=len(df), desc=desc, leave=False)

        debug_jsonl_path = os.path.join(self.cfg.save_dir, "debug_eval.jsonl")
        debug_log_path = os.path.join(self.cfg.save_dir, "debug_eval.log")

        # how many preview examples to log
        preview_examples_target = 2
        preview_examples_logged = 0

        # limits for logging / truncation
        PROMPT_SNIPPET_CHARS = 1200
        DECODED_INPUT_CHARS = 500
        INPUT_ID_SHOW = 200
        GENERATED_ID_SHOW = 300
        GENERATED_TEXT_SNIPPET = 2000

        def dbg_write_log(s: str):
            if self.cfg.debug:
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(s + "\n")
            logging.info(s)

        def dbg_write_jsonl(obj: dict):
            if self.cfg.debug:
                with open(debug_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        indices = list(range(len(df)))
        for i in range(0, len(df), self.cfg.eval_batch_size):
            batch_indices = indices[i : i + self.cfg.eval_batch_size]
            rows = [df.iloc[idx] for idx in batch_indices]

            prompts: list[str] = []
            corrects: list[str] = []
            row_meta: list[dict] = []

            for r in rows:
                idx = r.name

                if data_format == "mc":
                    try:
                        opts = literal_eval(r["options"])
                    except Exception as e:
                        total_errors += 1
                        logging.error(f"Skipping eval row idx={idx}: reason=options_parse_failed: {e}")
                        if self.cfg.debug:
                            dbg_write_log(f"[SKIP idx={idx}] options parse failed: {e}")
                            dbg_write_jsonl(
                                {
                                    "idx": idx,
                                    "reason": "options_parse_failed",
                                    "raw_options": r.get("options"),
                                }
                            )
                        continue

                    ai_val = None
                    if "answer_index" in r.index and pd.notna(r["answer_index"]):
                        try:
                            ai_val = int(r["answer_index"]) + 1
                        except Exception:
                            try:
                                ai_val = int(float(r["answer_index"])) + 1
                            except Exception:
                                ai_val = None
                    elif self.cfg.use_cot and "distill_answer" in r.index and pd.notna(r["distill_answer"]):
                        try:
                            ai_val = int(r["distill_answer"])
                        except Exception:
                            try:
                                ai_val = int(float(r["distill_answer"]))
                            except Exception:
                                ai_val = None

                    try:
                        prompt = self.format_prompt_qa(r, include_answer=False, new_format=False)
                    except Exception as e:
                        prompt = None
                        logging.error(f"Failed to format prompt for idx={idx}: {e}")
                        if self.cfg.debug:
                            dbg_write_log(f"[PROMPT FAIL idx={idx}] {e}")
                            dbg_write_jsonl(
                                {
                                    "idx": idx,
                                    "reason": "format_prompt_failed",
                                    "error": str(e),
                                }
                            )
                        total_errors += 1
                        continue

                    try:
                        single_tok = tokenizer(
                            prompt,
                            truncation=True,
                            max_length=self.cfg.max_input_length,
                            padding=False,
                            return_attention_mask=True,
                            add_special_tokens=False,
                        )
                        input_ids_list = (
                            single_tok["input_ids"]
                            if isinstance(single_tok["input_ids"], list)
                            else single_tok["input_ids"].tolist()
                        )
                        attn_mask_list = single_tok.get("attention_mask", None)
                        try:
                            decoded_snip = tokenizer.decode(
                                input_ids_list[:INPUT_ID_SHOW],
                                skip_special_tokens=True,
                            )
                        except Exception:
                            decoded_snip = ""
                    except Exception as e:
                        input_ids_list = []
                        attn_mask_list = None
                        decoded_snip = ""
                        logging.warning(f"Tokenization failed for idx={idx}: {e}")

                    if ai_val is None:
                        total_errors += 1
                        logging.error(
                            "Skipping eval row idx=%s: reason=No valid answer index, "
                            "answer_index=%s, distill_answer=%s, options=%s",
                            idx,
                            r.get("answer_index"),
                            r.get("distill_answer"),
                            opts,
                        )
                        if self.cfg.debug:
                            dbg_write_log(f"[SKIP idx={idx}] no valid answer index")
                            dbg_write_jsonl(
                                {
                                    "idx": idx,
                                    "reason": "no_answer_index",
                                    "raw_answer_index": r.get("answer_index"),
                                    "distill_answer": r.get("distill_answer"),
                                    "options": opts,
                                    "prompt": prompt,
                                    "prompt_snippet": prompt[:PROMPT_SNIPPET_CHARS],
                                    "input_ids": input_ids_list[:INPUT_ID_SHOW],
                                    "decoded_input_snippet": decoded_snip[:DECODED_INPUT_CHARS],
                                }
                            )
                        continue

                    if not (0 < int(ai_val) <= len(opts)):
                        total_errors += 1
                        logging.error(
                            "Skipping eval row idx=%s: reason=Invalid answer index, answer_index=%s, options_len=%s",
                            idx,
                            ai_val,
                            len(opts),
                        )
                        if self.cfg.debug:
                            dbg_write_log(f"[SKIP idx={idx}] invalid answer index {ai_val} (options_len={len(opts)})")
                            dbg_write_jsonl(
                                {
                                    "idx": idx,
                                    "reason": "invalid_answer_index",
                                    "raw_answer_index": r.get("answer_index"),
                                    "int_answer_index": ai_val,
                                    "options": opts,
                                    "prompt": prompt,
                                    "prompt_snippet": prompt[:PROMPT_SNIPPET_CHARS],
                                    "input_ids": input_ids_list[:INPUT_ID_SHOW],
                                    "decoded_input_snippet": decoded_snip[:DECODED_INPUT_CHARS],
                                }
                            )
                        continue

                    prompts.append(prompt)
                    corrects.append(str(int(ai_val)))
                    row_meta.append(
                        {
                            "index": idx,
                            "prompt": prompt,
                            "options": opts,
                            "expected": str(int(ai_val)),
                            "raw_answer_index": r.get("answer_index"),
                            "distill_answer": r.get("distill_answer"),
                            "question": r.get("question"),
                        }
                    )

                # ----- new jsonl format branch -----
                else:  # data_format == "jsonl"
                    if "question" not in r.index or "answer" not in r.index:
                        total_errors += 1
                        if self.cfg.debug:
                            dbg_write_log(f"[SKIP idx={idx}] missing 'question' or 'answer' column in jsonl row")
                            dbg_write_jsonl(
                                {
                                    "idx": idx,
                                    "reason": "missing_columns",
                                    "available_columns": list(r.index),
                                }
                            )
                        continue

                    try:
                        prompt = self.format_prompt_qa(r, include_answer=False, new_format=True)
                    except Exception as e:
                        logging.error(f"Failed to format prompt (jsonl) idx={idx}: {e}")
                        total_errors += 1
                        if self.cfg.debug:
                            dbg_write_log(f"[PROMPT FAIL idx={idx}] {e}")
                            dbg_write_jsonl(
                                {
                                    "idx": idx,
                                    "reason": "format_prompt_failed",
                                    "error": str(e),
                                }
                            )
                        continue

                    if prompt is None:
                        total_errors += 1
                        if self.cfg.debug:
                            dbg_write_log(f"[SKIP idx={idx}] prompt is None (jsonl)")
                        continue

                    gold = str(r["answer"]).strip()
                    if gold == "":
                        total_errors += 1
                        if self.cfg.debug:
                            dbg_write_log(f"[SKIP idx={idx}] empty gold answer (jsonl)")
                        continue

                    prompts.append(prompt)
                    corrects.append(gold)
                    row_meta.append(
                        {
                            "index": idx,
                            "prompt": prompt,
                            "options": None,
                            "expected": gold,
                            "raw_answer_index": r.get("answer_index"),
                            "distill_answer": r.get("distill_answer"),
                            "question": r.get("question"),
                        }
                    )

            if len(prompts) == 0:
                continue

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.max_input_length_eval,
                return_attention_mask=True,
            ).to(model.device)

            if self.cfg.debug:
                for j, meta in enumerate(row_meta[: min(3, len(row_meta))]):
                    try:
                        in_ids = inputs.input_ids[j].tolist()
                        attn_ids = inputs.attention_mask[j].tolist() if hasattr(inputs, "attention_mask") else None
                        dbg_write_log(
                            f"[BATCH INPUT idx={meta['index']}] prompt_snippet={prompts[j][:PROMPT_SNIPPET_CHARS]!r}"
                        )
                        dbg_write_log(
                            f"[BATCH INPUT idx={meta['index']}] "
                            f"token_ids(first{INPUT_ID_SHOW})={in_ids[:INPUT_ID_SHOW]}"
                        )
                        try:
                            decoded = tokenizer.decode(
                                in_ids[:INPUT_ID_SHOW],
                                skip_special_tokens=True,
                            )
                        except Exception:
                            decoded = ""
                        dbg_write_log(
                            f"[BATCH INPUT idx={meta['index']}] "
                            f"decoded(first{DECODED_INPUT_CHARS})={decoded[:DECODED_INPUT_CHARS]!r}"
                        )
                        dbg_write_jsonl(
                            {
                                "idx": meta["index"],
                                "reason": "prepare_for_generate",
                                "prompt": prompts[j],
                                "prompt_snippet": prompts[j][:PROMPT_SNIPPET_CHARS],
                                "input_ids": in_ids[:INPUT_ID_SHOW],
                                "attention_mask": attn_ids[:INPUT_ID_SHOW] if attn_ids is not None else None,
                                "expected": meta["expected"],
                                "options": meta["options"],
                                "data_format": data_format,
                            }
                        )
                    except Exception as e:
                        dbg_write_log(f"[DEBUG WARNING] failed to log batch input idx={meta['index']}: {e}")

            with torch.no_grad():
                try:
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask if hasattr(inputs, "attention_mask") else None,
                        max_new_tokens=self.cfg.max_new_tokens,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                except Exception as e:
                    # здесь различаем старую и новую семантику
                    if data_format == "jsonl":
                        total_errors += len(prompts)
                    else:  # mc
                        total_errors += 1
                    logging.error(f"Model.generate failed on batch starting idx {batch_indices[0]}: {e}")
                    if self.cfg.debug:
                        dbg_write_log(
                            f"[GEN ERROR] batch_start_idx={batch_indices[0]} error={repr(e)} data_format={data_format}"
                        )
                    continue

            responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            n = min(len(responses), len(corrects), len(row_meta), outputs.size(0))

            if not (len(responses) == len(corrects) == len(row_meta)):
                logging.warning(
                    "Length mismatch in eval batch: responses=%s, corrects=%s, meta=%s",
                    len(responses),
                    len(corrects),
                    len(row_meta),
                )
                if self.cfg.debug:
                    dbg_write_log(
                        f"[LENGTH MISMATCH] responses={len(responses)} corrects={len(corrects)} meta={len(row_meta)}"
                    )

            for k, (resp, corr, meta) in enumerate(zip(responses[:n], corrects[:n], row_meta[:n])):
                try:
                    matches = ANSWER_PATTERN.findall(resp)
                    gen = matches[-1].strip() if matches else None

                    gen_only_text = ""
                    try:
                        out_ids = outputs[k]
                        # input length per batch is the length of input_ids (before generation)
                        start_gen = inputs.input_ids.shape[1]
                        gen_only_ids = out_ids[start_gen:]
                        gen_only_text = tokenizer.decode(gen_only_ids, skip_special_tokens=True)
                    except Exception:
                        # if something went wrong — at least use the full resp
                        gen_only_text = resp

                    if preview_examples_logged < preview_examples_target:
                        prompt_for_log = meta.get("prompt", "")
                        question_for_log = meta.get("question", "")

                        logging.info(
                            f"\n===== PREVIEW {desc} example #{preview_examples_logged + 1} "
                            f"(row idx={meta.get('index')}) =====\n"
                            f"QUESTION:\n{question_for_log}\n\n"
                            f"PROMPT (what model sees):\n{prompt_for_log}\n\n"
                            f"MODEL FULL OUTPUT (decoded(outputs)):\n{resp}\n\n"
                            f"MODEL NEW TOKENS ONLY (answer part):\n{gen_only_text}\n\n"
                            f"PARSED ANSWER ([[...]]): {gen}\n"
                            f"GOLD ANSWER: {corr}\n"
                            "============================================"
                        )
                        preview_examples_logged += 1

                    if data_format == "jsonl" and gen is None:
                        total_errors += 1
                        if self.cfg.debug:
                            dbg_write_log(f"[PARSE MISS idx={meta['index']}] no answer marker found in generated text")
                            dbg_write_jsonl(
                                {
                                    "idx": meta["index"],
                                    "reason": "no_answer_marker",
                                    "generated_text_snippet": resp[:GENERATED_TEXT_SNIPPET],
                                    "regex_matches": matches,
                                    "expected": meta["expected"],
                                    "data_format": data_format,
                                }
                            )
                    else:
                        is_correct = gen == corr
                        total_correct += int(is_correct)
                        processed += 1

                        if self.cfg.debug:
                            try:
                                gen_ids = (
                                    outputs[k].tolist()[:GENERATED_ID_SHOW] if hasattr(outputs, "tolist") else None
                                )
                            except Exception:
                                gen_ids = None

                            dbg_prompt = meta.get("prompt") or self.format_prompt_qa(
                                df.loc[meta["index"]],
                                include_answer=False,
                                new_format=(data_format == "jsonl"),
                            )

                            dbg_obj = {
                                "idx": meta["index"],
                                "reason": "evaluated",
                                "prompt": dbg_prompt,
                                "prompt_snippet": dbg_prompt[:PROMPT_SNIPPET_CHARS],
                                "expected": meta["expected"],
                                "options": meta["options"],
                                "generated_text": resp[:GENERATED_TEXT_SNIPPET],
                                "generated_ids": gen_ids,
                                "regex_matches": matches,
                                "is_correct": is_correct,
                                "data_format": data_format,
                            }
                            dbg_write_jsonl(dbg_obj)

                            if not is_correct:
                                dbg_write_log(
                                    f"[MISMATCH idx={meta['index']} format={data_format}] "
                                    f"expected={meta['expected']} regex_matches={matches}"
                                )

                except Exception as e:
                    total_errors += 1
                    logging.error(f"Eval parse error idx={meta.get('index')}: {e}")
                    if self.cfg.debug:
                        dbg_write_log(
                            f"[PARSE ERROR] idx={meta.get('index')} error={repr(e)} resp_snippet={resp[:1000]!r}"
                        )
                pbar.update(1)

            if processed:
                pbar.set_postfix(
                    {
                        "acc": f"{(total_correct / processed) * 100:.2f}%",
                        "errors": total_errors,
                    }
                )
            else:
                pbar.set_postfix({"acc": "0%", "errors": total_errors})

        pbar.close()
        accuracy = total_correct / (processed if processed else 1.0)
        self._add_result(epoch=epoch, accuracy=accuracy, desc=desc)
        return accuracy

    # training loop
    def train(self, save=False):
        from torch.amp import GradScaler, autocast
        from torch.optim import AdamW

        self.load_model_and_tokenizer()

        train_ds, val_ds, test_ds, train_df, val_df, test_df = self.prepare_datasets(
            self.cfg.train_path, self.cfg.valid_path, self.cfg.test_path
        )

        if self.cfg.use_cot:
            cot_column = None
            if "distill_response" in train_df.columns and train_df["distill_response"].notna().any():
                cot_column = "distill_response"
            elif "reasoning" in train_df.columns and train_df["reasoning"].notna().any():
                cot_column = "reasoning"

            if cot_column is not None:
                logging.info(f"CoT from column: '{cot_column}' (train_df)")
            else:
                logging.warning(
                    "empty CoT column: no 'distill_response' or 'reasoning' column with data found in train_df"
                )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False, pad_to_multiple_of=8)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=self.cfg.num_workers,
        )

        optimizer = AdamW(self.model.parameters(), lr=self.cfg.lr, fused=True)
        scaler = GradScaler(enabled=False)

        if self.cfg.run_eval_on_start:
            logging.info("Evaluating model before training")

            if test_df is not None and len(test_df) > 0:
                test_balanced_df = test_df
                logging.info(f"Using in-memory test_df for initial eval: {len(test_balanced_df)} rows")
            else:
                # Фолбэк на старое поведение, если вдруг test_df пустой
                if self.cfg.test_balanced_path.endswith((".jsonl", ".json")):
                    test_balanced_df = pd.read_json(self.cfg.test_balanced_path, lines=True)
                else:
                    test_balanced_df = pd.read_csv(self.cfg.test_balanced_path, sep="\t")
                logging.info(
                    f"Loaded test_balanced_df from path {self.cfg.test_balanced_path}: {len(test_balanced_df)} rows"
                )

            test_acc = self.evaluate_qa(
                self.model,
                test_balanced_df,
                self.tokenizer,
                0,
                desc="test_balanced",
                data_format="jsonl",  # для gsm8k-jsonl
            )
            logging.info(f"TEST BALANCED QA Accuracy: {test_acc * 100:.2f}%")

        for epoch in range(self.cfg.epochs):
            self.model.train()
            total_train_loss = 0.0
            total_train_acc = 0.0
            total_samples = 0
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{self.cfg.epochs}")

            for batch_idx, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if batch_idx == 0:
                    sample = self.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
                    logging.info(f"Training batch input ids shape: {batch['input_ids'].shape}")
                    logging.info(f"\nTraining Sample:\n{sample}")

                with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16):
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.cfg.gradient_accumulation
                    accuracy = calculate_accuracy(outputs.logits, batch["input_ids"], self.tokenizer)

                scaler.scale(loss).backward()

                if (batch_idx + 1) % self.cfg.gradient_accumulation == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                bs = batch["input_ids"].size(0)
                total_train_loss += loss.item() * bs
                total_train_acc += accuracy.item() * bs
                total_samples += bs

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{(total_train_loss / total_samples) * self.cfg.gradient_accumulation:.3f}",
                        "acc": f"{(total_train_acc / total_samples) * 100:.2f}%",
                        "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                    }
                )

                if (batch_idx + 1) % self.cfg.log_interval == 0:
                    logging.info(
                        f"Epoch {epoch + 1}, Batch {batch_idx + 1}: Loss = {(total_train_loss / total_samples) * self.cfg.gradient_accumulation:.3f}, Acc = {(total_train_acc / total_samples) * 100:.2f}%"
                    )
                    total_train_loss = 0.0
                    total_train_acc = 0.0
                    total_samples = 0

            pbar.close()
            logging.info(f"Epoch {epoch + 1} completed.")

            if self.cfg.eval_validation_period != 0 and (
                (epoch + 1) == self.cfg.epochs or (epoch + 1) % self.cfg.eval_validation_period == 0
            ):
                # validation
                val_acc = self.evaluate_qa(self.model, val_df, self.tokenizer, epoch + 1, desc="validation")
                logging.info(f"Validation QA Accuracy: {val_acc * 100:.2f}%")

            skip_eval_test = self.cfg.eval_test_period == 0
            is_last_epoch = (epoch + 1) == self.cfg.epochs
            if type(self.cfg.eval_test_period) is list:
                epoch_matches_eval_test_period = (epoch + 1) in self.cfg.eval_test_period
            else:
                epoch_matches_eval_test_period = (
                    self.cfg.eval_test_period != 0 and (epoch + 1) % self.cfg.eval_test_period == 0
                )
            if not skip_eval_test and (is_last_epoch or epoch_matches_eval_test_period):
                # balanced test (from path)
                if self.cfg.test_balanced_path.endswith((".jsonl", ".json")):
                    test_balanced_df = pd.read_json(self.cfg.test_balanced_path, lines=True)
                else:
                    test_balanced_df = pd.read_csv(self.cfg.test_balanced_path, sep="\t")

                test_acc = self.evaluate_qa(
                    self.model,
                    test_balanced_df,
                    self.tokenizer,
                    epoch + 1,
                    desc="test_balanced",
                    data_format="jsonl",
                )
                logging.info(f"TEST BALANCED QA Accuracy: {test_acc * 100:.2f}%")

            # save checkpoint (disabled by default, enable if you want)
            # epoch_dir = os.path.join(self.cfg.save_dir, f"epoch_{epoch+1}")
            # os.makedirs(epoch_dir, exist_ok=True)
            # self.model.save_pretrained(epoch_dir)
            # self.tokenizer.save_pretrained(epoch_dir)
            # logging.info(f"Model saved to {epoch_dir}")
        self.model.save_pretrained(self.cfg.save_dir)
        self.tokenizer.save_pretrained(self.cfg.save_dir)
        logging.info(f"Model saved to {self.cfg.save_dir}")
