import os, ast, json, re, logging, math, time
from concurrent import futures

import pandas as pd
from tqdm import tqdm

from core.utils.openrouter import openrouter
from core.utils.chunker import chunker

from core.prompts.mmlu_single_token_answer import (
    single_token_sys_prompt,
    single_token_answer_prompt,
)

from core.prompts.mmlu_branches_aug import (
    option_ids,
    explain_sys_prompt,
    explain_user_prompt,
    error_review_sys_prompt,
    error_review_messages,
)

ALL_LETTERS = [chr(c) for c in range(ord("A"), ord("Z")+1)]
logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


# ------------ utils ------------
def letters_for(n: int):
    n = max(0, min(int(n), 26))
    return ALL_LETTERS[:n]

def parse_options(s):
    lst = ast.literal_eval(s)
    return list(map(str, lst))

def norm_letter_dyn(x, letters):
    s = ("" if x is None else str(x)).strip().upper()
    if s in letters:
        return s
    if s.isdigit():
        i = int(s)
        if 0 <= i < len(letters):
            return letters[i]
        if 0 <= i-1 < len(letters):
            return letters[i-1]
    return ""

def _subject_from_row(row_dict: dict) -> str | None:
    return (row_dict.get("base_cluster") or row_dict.get("category") or row_dict.get("subject") or row_dict.get("src") or "").strip() or None

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
def ask_mcq_once(question: str,
                 choices: list[str],
                 gold_letter: str,
                 model: str,
                 max_tokens: int,
                 subject: str | None,
                 temperature: float = 0) -> dict:
    letters = letters_for(len(choices))
    sys_prompt = single_token_sys_prompt(subject)
    user_prompt = single_token_answer_prompt(question, choices)


    completion = openrouter.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={ "include_reasoning": True }
    )
    msg = completion.choices[0].message
    content = msg.content or ""
    reasoning_text = getattr(msg, "reasoning", None)

    ans_letter = _extract_letter_from_text(content, letters)
    if not ans_letter:
        ans_letter = (content.strip()[:1] or "").upper()
        if ans_letter not in letters:
            ans_letter = ""

    is_correct = (ans_letter.upper() == (gold_letter or "").upper())

    return {
        "answer": ans_letter,
        "is_correct": is_correct,
        "thinking": reasoning_text or "",
        "raw_response": content
    }

def _branch_a(q, choices, gold, model, max_tokens, subject, temperature):
    return ask_mcq_once(q, choices, gold, model=model, max_tokens=max_tokens, subject=subject, temperature=temperature)


# ------------ branch B ------------
def ask_mcq_explain(question: str,
                    choices: list[str],
                    gold_letter: str,
                    model: str,
                    max_tokens: int,
                    subject: str | None,
                    temperature: float = 0) -> dict:
    letters = letters_for(len(choices))
    sys_prompt = explain_sys_prompt(subject)
    user_prompt = explain_user_prompt(question, choices, gold_letter)

    completion = openrouter.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={ "include_reasoning": True }
    )
    msg = completion.choices[0].message
    content = msg.content or ""
    reasoning_text = getattr(msg, "reasoning", None) or ""

    return {
        "thinking": reasoning_text,
        "raw_response": content
    }

def _branch_b(q, choices, gold, model, max_tokens, subject, temperature):
    return ask_mcq_explain(q, choices, gold, model=model, max_tokens=max_tokens, subject=subject, temperature=temperature)


# ------------ branch C ------------
def ask_mcq_error_review(question: str,
                         choices: list[str],
                         gold_letter: str,
                         model_letter_from_a: str,
                         prev_reasoning_from_a: str,
                         model: str,
                         max_tokens: int,
                         subject: str | None,
                         temperature: float = 0) -> dict:
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
        extra_body={ "include_reasoning": True }
    )

    msg = completion.choices[0].message
    content = msg.content or ""
    reasoning_text = getattr(msg, "reasoning", None) or ""

    return {
        "gold": gold_letter,
        "preivous_answer": (model_letter_from_a or "").upper(),
        "thinking": reasoning_text,
        "raw_response": content
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
def _load_incorrect_from_branch_a(a_jsonl_path: str, expected_model: str | None) -> dict[int, dict]:
    bad: dict[int, dict] = {}
    with open(a_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            inp = rec.get("input") or {}
            out = rec.get("output") or {}
            if "error" in out:
                continue
            if expected_model is not None and (inp.get("model") != expected_model):
                continue
            row_id = inp.get("row_id")
            if row_id is None:
                continue
            gold = (inp.get("gold") or "").strip().upper()
            ans = (out.get("answer") or "").strip().upper()
            is_correct = out.get("is_correct")
            if is_correct is None:
                is_correct = (ans == gold)
            if not is_correct:
                bad[int(row_id)] = {
                    "preivous_answer": ans,
                    "thinking": out.get("thinking") or "",
                }
    return bad


def _load_and_clean_existing(out_jsonl: str) -> set[int]:
    if not os.path.exists(out_jsonl):
        return set()
    
    valid_ids = set()
    valid_lines = []
    
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
                # Check if output has error
                if "error" not in rec.get("output", {}):
                    rid = rec.get("input", {}).get("question_id")
                    if rid is not None:
                        valid_ids.add(int(rid))
                    valid_lines.append(line)
            except Exception:
                pass
    
    # Rewrite file with only valid lines
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for line in valid_lines:
            f.write(line + "\n")
            
    return valid_ids


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
            out = _branch_c(question, choices, gold_letter, model, max_tokens, subject, prev_answer, prev_reasoning, temperature)
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
        record_in["model_answer_from_A"] = (prev_answer or "")

    return row_id, record_in, out


def synth_on_dataset(
    in_filename: str,
    out_jsonl: str,
    model: str,
    max_tokens: int,
    dump_every: int,
    limit: int | None,
    branch: str,
    chunk_size: int,
    a_jsonl_path: str | None,
    temperature: float = 0, # [warning]: temperature for all branches
):
    assert branch in {"A", "B", "C"}
    if branch == "C":
        assert a_jsonl_path and os.path.exists(a_jsonl_path), "Branch C requires a valid path to branch-A results (a_jsonl_path)."

    df = pd.read_csv(in_filename, sep="\t", dtype=str, keep_default_na=False)
    total_rows = len(df) if limit is None else min(len(df), int(limit))
    total_chunks = max(1, math.ceil(total_rows / max(1, chunk_size)))

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    existing_ids = _load_and_clean_existing(out_jsonl)
    logging.warning(f"Found {len(existing_ids)} valid records in {out_jsonl}. Errors removed.")

    # pre-load A-incorrects for branch C
    a_incorrect_map: dict[int, dict] = {}
    ids_for_c: set[int] = set()
    if branch == "C":
        a_incorrect_map = _load_incorrect_from_branch_a(a_jsonl_path, expected_model=model)
        ids_for_c = set(a_incorrect_map.keys())

    written = 0
    stop = False

    with open(out_jsonl, "a", encoding="utf-8") as f, futures.ThreadPoolExecutor(max_workers=chunk_size) as pool:
        for chunk_idx, chunk in tqdm(enumerate(chunker(df, chunk_size)), total=total_chunks, desc=f"Synth {branch}"):
            if stop:
                break

            args_list = []
            for index, row in chunk.iterrows():
                if row["question_id"] in existing_ids:
                    continue

                if limit is not None and written >= limit:
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

                gold_letter = (
                    norm_letter_dyn(row_dict.get("answer"), letters)
                    or norm_letter_dyn(row_dict.get("answer_index"), letters)
                )
                if not gold_letter:
                    continue

                prev_ans = None
                prev_thinking = None
                if branch == "C":
                    if index not in ids_for_c:
                        continue
                    prev_ans = a_incorrect_map[index].get("model_answer")
                    prev_thinking = a_incorrect_map[index].get("thinking")

                args_list.append((index, question_id, q, choices, gold_letter, model, max_tokens, branch, subject, prev_ans, prev_thinking, temperature))

            if not args_list:
                continue

            results = list(pool.map(_run_job, args_list))

            for row_id, record_in, record_out in results:
                f.write(json.dumps({"input": record_in, "output": record_out}, ensure_ascii=False) + "\n")
                written += 1
                if dump_every > 0 and (written % dump_every == 0):
                    f.flush()

    print(f"Saved to {out_jsonl}. Rows considered: {len(df)}; written: {written}; branch={branch}; model={model}.")
    return out_jsonl
