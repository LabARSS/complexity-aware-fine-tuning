import ast
import logging
import math
import os
import re
from concurrent import futures

import pandas as pd
from tqdm import tqdm

from core.prompts.mmlu_branches_aug import (
    error_review_messages,
    error_review_sys_prompt,
    explain_sys_prompt,
    explain_user_prompt,
)
from core.prompts.mmlu_single_token_answer import (
    single_token_answer_prompt,
    single_token_sys_prompt,
)
from core.utils.chunker import chunker
from core.utils.openrouter import openrouter

ALL_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


# ------------ utils ------------
def letters_for(n: int):
    n = max(0, min(int(n), 26))
    return ALL_LETTERS[:n]


def parse_options(s):
    if isinstance(s, list):
        return list(map(str, s))
    try:
        lst = ast.literal_eval(str(s))
        return list(map(str, lst))
    except:
        return []


def norm_letter_dyn(x, letters):
    s = ("" if x is None else str(x)).strip().upper()
    if s in letters:
        return s
    if s.isdigit():
        i = int(s)
        if 0 <= i < len(letters):
            return letters[i]
        if 0 <= i - 1 < len(letters):
            return letters[i - 1]
    return ""


def _subject_from_row(row_dict: dict) -> str | None:
    return (
        row_dict.get("base_cluster") or row_dict.get("category") or row_dict.get("subject") or row_dict.get("src") or ""
    ).strip() or None


def _extract_letter_from_text(txt: str, letters: list[str]) -> str:
    # extract first allow letter from text
    t = (txt or "").strip()
    t = re.sub(r"^```(?:[a-zA-Z]+)?\s*|\s*```$", "", t, flags=re.S)
    for ch in t:
        if ch.upper() in letters:
            return ch.upper()
    m = re.search(r"(?<!\d)(\d{1,2})(?!\d)", t)
    if m:
        return norm_letter_dyn(m.group(1), letters)
    return ""


# ------------ branch A ------------
def ask_mcq_once(
    question: str,
    choices: list[str],
    gold_letter: str,
    model: str,
    max_tokens: int,
    subject: str | None,
    temperature: float = 0,
) -> dict:
    letters = letters_for(len(choices))
    sys_prompt = single_token_sys_prompt(subject)
    user_prompt = single_token_answer_prompt(question, choices)

    completion = openrouter.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"include_reasoning": True},
    )
    msg = completion.choices[0].message
    content = msg.content or ""
    reasoning_text = getattr(msg, "reasoning", None)

    ans_letter = _extract_letter_from_text(content, letters)
    if not ans_letter:
        ans_letter = (content.strip()[:1] or "").upper()
        if ans_letter not in letters:
            ans_letter = ""

    is_correct = ans_letter.upper() == (gold_letter or "").upper()

    return {"answer": ans_letter, "is_correct": is_correct, "thinking": reasoning_text or "", "raw_response": content}


def _branch_a(q, choices, gold, model, max_tokens, subject, temperature):
    return ask_mcq_once(q, choices, gold, model=model, max_tokens=max_tokens, subject=subject, temperature=temperature)


# ------------ branch B ------------
def ask_mcq_explain(
    question: str,
    choices: list[str],
    gold_letter: str,
    model: str,
    max_tokens: int,
    subject: str | None,
    temperature: float = 0,
) -> dict:
    letters = letters_for(len(choices))
    sys_prompt = explain_sys_prompt(subject)
    user_prompt = explain_user_prompt(question, choices, gold_letter)

    completion = openrouter.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"include_reasoning": True},
    )
    msg = completion.choices[0].message
    content = msg.content or ""
    reasoning_text = getattr(msg, "reasoning", None) or ""

    return {"thinking": reasoning_text, "raw_response": content}


def _branch_b(q, choices, gold, model, max_tokens, subject, temperature):
    return ask_mcq_explain(
        q, choices, gold, model=model, max_tokens=max_tokens, subject=subject, temperature=temperature
    )


# ------------ branch C ------------
def ask_mcq_error_review(
    question: str,
    choices: list[str],
    gold_letter: str,
    model_letter_from_a: str,
    prev_reasoning_from_a: str,
    model: str,
    max_tokens: int,
    subject: str | None,
    temperature: float = 0,
) -> dict:
    letters = letters_for(len(choices))
    sys_prompt = error_review_sys_prompt(subject)
    extra_msgs = error_review_messages(
        question=question,
        options=choices,
        model_letter=model_letter_from_a or "",
        gold_letter=gold_letter,
        previous_reasoning=prev_reasoning_from_a or "",
    )

    completion = openrouter.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys_prompt}] + extra_msgs,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"include_reasoning": True},
    )

    msg = completion.choices[0].message
    content = msg.content or ""
    reasoning_text = getattr(msg, "reasoning", None) or ""

    return {
        "gold": gold_letter,
        "preivous_answer": (model_letter_from_a or "").upper(),
        "thinking": reasoning_text,
        "raw_response": content,
    }


def _branch_c(q, choices, gold, model, max_tokens, subject, prev_answer, prev_reasoning, temperature):
    return ask_mcq_error_review(
        question=q,
        choices=choices,
        gold_letter=gold,
        model_letter_from_a=prev_answer,
        prev_reasoning_from_a=prev_reasoning,
        model=model,
        max_tokens=max_tokens,
        subject=subject,
        temperature=temperature,
    )


# ------------ helpers for branch C ------------
def _load_incorrect_from_branch_a(a_parquet_path: str, expected_model: str | None) -> dict[int, dict]:
    if not a_parquet_path or not os.path.exists(a_parquet_path):
        return {}
    
    try:
        df = pd.read_parquet(a_parquet_path)
    except Exception:
        return {}

    bad = {}
    for _, row in df.iterrows():
        inp, out = row["input"], row["output"]
        if "error" in out:
            continue
        if expected_model and inp.get("model") != expected_model:
            continue
            
        # Check correctness (prefer explicit flag, fallback to string comparison)
        is_correct = out.get("is_correct")
        if is_correct is None:
            is_correct = (out.get("answer") or "").strip().upper() == (inp.get("gold") or "").strip().upper()
            
        if not is_correct:
            bad[int(inp["question_id"])] = {
                "model_answer": out.get("answer"),
                "thinking": out.get("thinking", ""),
            }
    return bad


def _load_existing_ids(out_parquet: str) -> set[int]:
    if not os.path.exists(out_parquet):
        return set()
    try:
        df = pd.read_parquet(out_parquet, columns=["input", "output"])
        return {
            int(row["input"]["question_id"])
            for _, row in df.iterrows()
            if "error" not in row["output"]
        }
    except Exception:
        return set()


# ------------ dataset -------------
def _run_job(job):
    (
        row_id,
        question_id,
        question,
        choices,
        gold_letter,
        model,
        max_tokens,
        branch,
        subject,
        prev_answer,
        prev_reasoning,
        temperature,
    ) = job

    try:
        if branch == "A":
            out = _branch_a(question, choices, gold_letter, model, max_tokens, subject, temperature)
        elif branch == "B":
            out = _branch_b(question, choices, gold_letter, model, max_tokens, subject, temperature)
        else:
            out = _branch_c(
                question, choices, gold_letter, model, max_tokens, subject, prev_answer, prev_reasoning, temperature
            )
    except Exception as e:
        logging.warning(f"[idx={row_id}] error: {e}")
        out = {"error": str(e)}

    letters = letters_for(len(choices))
    record_in = {
        "question_id": question_id,
        "subject": subject or "",
        "question": question,
        "options": {letters[i]: choices[i] for i in range(len(choices))},
        "gold": (gold_letter or "").upper(),
        "model": model,
        "branch": branch,
    }
    if branch == "C":
        record_in["model_answer_from_A"] = prev_answer or ""

    return row_id, record_in, out


def synth_on_dataset(
    in_filename: str,
    out_filename: str,
    model: str,
    max_tokens: int,
    dump_every: int,
    limit: int | None,
    branch: str,
    chunk_size: int,
    a_file_path: str | None,
    temperature: float = 0,
):
    assert branch in {"A", "B", "C"}
    if branch == "C":
        assert a_file_path and os.path.exists(a_file_path), (
            "Branch C requires a valid path to branch-A parquet results (a_file_path)."
        )

    # Read input dataset (TSV/CSV)
    df = pd.read_csv(in_filename, sep="\t", dtype=str, keep_default_na=False)
    total_rows = len(df) if limit is None else min(len(df), int(limit))
    total_chunks = max(1, math.ceil(total_rows / max(1, chunk_size)))

    os.makedirs(os.path.dirname(out_filename) or ".", exist_ok=True)

    # Load existing progress
    existing_ids = _load_existing_ids(out_filename)
    logging.warning(f"Found {len(existing_ids)} valid records in {out_filename}.")

    # Pre-load A-incorrects for branch C
    a_incorrect_map: dict[int, dict] = {}
    ids_for_c: set[int] = set()
    if branch == "C":
        a_incorrect_map = _load_incorrect_from_branch_a(a_file_path, expected_model=model)
        ids_for_c = set(a_incorrect_map.keys())
        logging.info(f"Loaded {len(ids_for_c)} incorrect answers from Branch A for processing.")

    written = 0
    stop = False
    buffer = []

    with futures.ThreadPoolExecutor(max_workers=chunk_size) as pool:
        for chunk_idx, chunk in tqdm(enumerate(chunker(df, chunk_size)), total=total_chunks, desc=f"Synth {branch}"):
            if stop:
                break

            args_list = []
            for index, row in chunk.iterrows():
                # Check if already processed
                qid = row.get("question_id")
                if qid and int(qid) in existing_ids:
                    continue

                if limit is not None and (len(existing_ids) + written) >= limit:
                    stop = True
                    break

                if index >= total_rows:
                    stop = True
                    break

                row_dict = row.to_dict()
                subject = _subject_from_row(row_dict)

                q = (row_dict.get("question") or "").strip()
                choices = parse_options(row_dict.get("options") or "[]")
                letters = letters_for(len(choices))
                if len(letters) < 2 or not q:
                    continue

                question_id = row_dict.get("question_id")

                gold_letter = norm_letter_dyn(row_dict.get("answer"), letters) or norm_letter_dyn(
                    row_dict.get("answer_index"), letters
                )
                if not gold_letter:
                    continue

                prev_ans = None
                prev_thinking = None
                if branch == "C":
                    # Only process rows where Branch A failed
                    # Use question_id for lookup, NOT the dataframe index
                    try:
                        qid_int = int(question_id)
                    except (ValueError, TypeError):
                        continue

                    if qid_int not in ids_for_c:
                        continue
                    prev_ans = a_incorrect_map[qid_int].get("model_answer")
                    prev_thinking = a_incorrect_map[qid_int].get("thinking")

                args_list.append(
                    (
                        index,
                        question_id,
                        q,
                        choices,
                        gold_letter,
                        model,
                        max_tokens,
                        branch,
                        subject,
                        prev_ans,
                        prev_thinking,
                        temperature,
                    )
                )

            if not args_list:
                continue

            results = list(pool.map(_run_job, args_list))

            for row_id, record_in, record_out in results:
                buffer.append({"input": record_in, "output": record_out})
                written += 1
            
            # Dump to parquet periodically
            if dump_every > 0 and len(buffer) >= dump_every:
                try:
                    new_df = pd.DataFrame(buffer)
                    if os.path.exists(out_filename):
                        existing_df = pd.read_parquet(out_filename)
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    else:
                        combined_df = new_df
                    
                    combined_df.to_parquet(out_filename, index=False)
                    buffer = [] # Clear buffer after successful write
                except Exception as e:
                    logging.error(f"Failed to write parquet batch: {e}")

    # Final flush
    if buffer:
        try:
            new_df = pd.DataFrame(buffer)
            if os.path.exists(out_filename):
                existing_df = pd.read_parquet(out_filename)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            combined_df.to_parquet(out_filename, index=False)
        except Exception as e:
            logging.error(f"Failed to write final parquet batch: {e}")

    print(f"Saved to {out_filename}. Rows considered: {len(df)}; written: {written}; branch={branch}; model={model}.")
    return out_filename
