import argparse
import importlib
from typing import Callable, Tuple, Optional, List

from core.utils.device import  DEVICE, move_batch_to_device

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import core.prompts.mmlu_single_token_answer as mmlu_single
import core.prompts.mmlu_cot_answer as mmlu_cot
from core.complexity_estimation.entropy.runner import EntropyConfig, run_entropy_estimation
from core.complexity_estimation.entropy.strategies import (
    make_prompt_builder, single_token_answer_extractor, cot_answer_extractor
)

# dotted import (для адапретов датасета (бэклог))
def _load_callable(dotted: str) -> Callable:
    mod, func = dotted.split(":")
    return getattr(importlib.import_module(mod), func)

# Фоллбек без адаптера
def _normalize_letter(ans: str) -> str:
    ans = (ans or "").strip().upper()
    return ans[:1]  # берем первую букву

def build_row_funcs_from_columns(
    subject_col: str,
    question_col: str,
    options_col: str,
    answer_index_col: Optional[str],
    answer_letter_col: Optional[str],
    index_base: int,
    letters: List[str],
):
    letters_u = [L.strip().upper() for L in letters]
    letter_to_idx = {L: i for i, L in enumerate(letters_u)}

    def get_subject_from_row(row):   # -> str
        return str(row[subject_col])

    def get_question_from_row(row):
        return str(row[question_col])

    def get_options_from_row(row):
        return str(row[options_col])

    if answer_letter_col:
        # Сравниваем буквы напрямую
        def check_answer_correct(row, answer: str) -> bool:
            gt = str(row[answer_letter_col]).strip().upper()[:1]
            pred = _normalize_letter(answer)
            return pred == gt
    elif answer_index_col:
        # Сопоставляем букву с индексом (A->0, B->1, ...) и сравниваем с answer_index - index_base
        def check_answer_correct(row, answer: str) -> bool:
            pred = _normalize_letter(answer)
            if pred not in letter_to_idx:
                return False
            pred_idx = letter_to_idx[pred]
            try:
                gt_idx = int(row[answer_index_col]) - int(index_base)
            except Exception:
                return False
            return pred_idx == gt_idx
    else:
        raise ValueError("Provide either --answer-index-col or --answer-letter-col")

    return (get_subject_from_row, get_question_from_row, get_options_from_row, check_answer_correct)

# вариант адаптера
def build_row_funcs_from_adapter(dataset_adapter: str):
    adapter = _load_callable(dataset_adapter)
    return (adapter.get_subject_from_row,
            adapter.get_question_from_row,
            adapter.get_options_from_row,
            adapter.check_answer_correct)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_file", required=True)
    ap.add_argument("--out", dest="out_file", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["single_token", "cot"], required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dump-every", type=int, default=1000)
    ap.add_argument("--max-new-tokens", type=int, default=None)

    ap.add_argument("--dataset-adapter", type=str, default=None,
                    help="Dotted path to adapter with row functions "
                         "(e.g., core.datasets.mmlu:adapter). Optional if you use column flags below.")

    # без адаптера
    ap.add_argument("--subject-col", type=str, default=None, help="e.g., 'src' or 'subject'")
    ap.add_argument("--question-col", type=str, default=None, help="e.g., 'question'")
    ap.add_argument("--options-col", type=str, default=None, help="e.g., 'options'")
    ap.add_argument("--answer-index-col", type=str, default=None, help="e.g., 'answer_index'")
    ap.add_argument("--answer-letter-col", type=str, default=None, help="e.g., 'answer' with A/B/C/D")
    ap.add_argument("--index-base", type=int, default=0, choices=[0, 1],
                    help="Use 0 if answer_index is 0-based, 1 if 1-based.")
    ap.add_argument("--letters", type=str, default="A,B,C,D",
                    help="Ordered labels for options, comma-separated. Default: A,B,C,D")

    # Смена промптов
    ap.add_argument("--sys-prompt-fn", type=str, default=None)
    ap.add_argument("--user-prompt-fn", type=str, default=None)

    # Эмбеддинги
    ap.add_argument("--compute-input-emb", action="store_true")
    ap.add_argument("--compute-think-answer-emb", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu" # -- Изменить на DEVICE (core.utils.device.DEVICE)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto" if device == "cuda" else None)
  
    if args.dataset_adapter:
        get_subject_from_row, get_question_from_row, get_options_from_row, check_answer_correct = \
            build_row_funcs_from_adapter(args.dataset_adapter)
    else:
        if not (args.subject_col and args.question_col and args.options_col and (args.answer_index_col or args.answer_letter_col)):
            raise SystemExit("Adapterless mode: please provide --subject-col, --question-col, --options-col "
                             "and one of (--answer-index-col or --answer-letter-col)")
        letters = [s.strip() for s in args.letters.split(",")]
        get_subject_from_row, get_question_from_row, get_options_from_row, check_answer_correct = \
            build_row_funcs_from_columns(
                subject_col=args.subject_col,
                question_col=args.question_col,
                options_col=args.options_col,
                answer_index_col=args.answer_index_col,
                answer_letter_col=args.answer_letter_col,
                index_base=args.index_base,
                letters=letters,
            )

    if args.mode == "single_token":
        sys_fn = _load_callable(args.sys_prompt_fn) if args.sys_prompt_fn else mmlu_single.single_token_sys_prompt
        user_fn = _load_callable(args.user_prompt_fn) if args.user_prompt_fn else mmlu_single.single_token_answer_prompt
        prompt_builder = make_prompt_builder(sys_fn, user_fn,
                                             get_subject_from_row, get_question_from_row, get_options_from_row)
        cfg = EntropyConfig(
            mode="single_token",
            max_new_tokens=args.max_new_tokens or 1,
            entropy_reducer="last_step",
            prompt_builder=prompt_builder,
            answer_extractor=single_token_answer_extractor,
            check_answer_correct=check_answer_correct,
            batch_size=args.batch_size,
            dump_every=args.dump_every,
            compute_input_embeddings=args.compute_input_emb,
        )
    else:
        sys_fn = _load_callable(args.sys_prompt_fn) if args.sys_prompt_fn else mmlu_cot.cot_sys_prompt
        user_fn = _load_callable(args.user_prompt_fn) if args.user_prompt_fn else mmlu_cot.cot_answer_prompt
        prompt_builder = make_prompt_builder(sys_fn, user_fn,
                                             get_subject_from_row, get_question_from_row, get_options_from_row)
        cfg = EntropyConfig(
            mode="cot",
            max_new_tokens=args.max_new_tokens or 1024,
            entropy_reducer="sequence",
            prompt_builder=prompt_builder,
            answer_extractor=cot_answer_extractor,
            check_answer_correct=check_answer_correct,
            batch_size=args.batch_size,
            dump_every=args.dump_every,
            compute_input_embeddings=args.compute_input_emb,
            compute_think_answer_embeddings=args.compute_think_answer_emb,
        )

    run_entropy_estimation(
        in_filename=args.in_file,
        out_filename=args.out_file,
        model=mdl,
        tokenizer=tok,
        cfg=cfg,
        get_subject_from_row=get_subject_from_row,
        get_question_from_row=get_question_from_row,
        get_options_from_row=get_options_from_row,
        resume=True,
        num_proc=4,
    )

if __name__ == "__main__":
    main()
