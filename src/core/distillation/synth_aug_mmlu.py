import os, ast, json, re, logging, math
from concurrent import futures

import pandas as pd
from tqdm import tqdm

from core.utils.openrouter import openrouter
from core.utils.chunker import chunker

from core.prompts.mmlu_branches_aug import (
    render_mc_prompt,
    render_mc_prompt_b,
    render_mc_prompt_c_review,
    _schema_answer_only,
    _schema_explanations_only,
    _schema_c_review,
)

ALL_LETTERS = [chr(c) for c in range(ord("A"), ord("Z")+1)]

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

# -------------------- utils --------------------
def letters_for(n: int):
    n = max(0, min(int(n), 26))
    return ALL_LETTERS[:n]

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

def parse_options(s):
    try:
        lst = ast.literal_eval(s)
        return list(map(str, lst))
    except Exception:
        s = (s or "").strip().strip("[]")
        parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
        return [p for p in parts if p]

def _coerce_json(txt: str) -> dict:
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", (txt or "").strip(), flags=re.S)
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        s = s[i:j+1]
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return json.loads(s)

# -------------------- branch A --------------------
def ask_mcq_once(question: str, choices: list[str], gold_letter: str,
                 model: str, max_tokens: int) -> dict:
    letters = letters_for(len(choices))
    sys_prompt, user_prompt = render_mc_prompt(question, choices, letters)

    completion = openrouter.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
        extra_body={
            "provider": {"require_parameters": True},
            "response_format": {"type": "json_schema", "json_schema": _schema_answer_only(letters)},
            "include_reasoning": True,
            "reasoning": {"enabled": True},
        }
    )

    msg = completion.choices[0].message
    txt = msg.content or ""
    reasoning_text = getattr(msg, "reasoning", None)

    try:
        j = json.loads(txt)
    except Exception:
        try:
            j = _coerce_json(txt)
        except Exception:
            logging.warning("JSON parse failed; returning empty object")
            j = {}

    ans_letter = norm_letter_dyn(j.get("answer"), letters)
    is_correct = (ans_letter == gold_letter)

    return {
        "letters": letters,
        "options": {letters[i]: choices[i] for i in range(len(choices))},
        "gold": gold_letter,
        "answer": ans_letter,
        "is_correct": is_correct,
        "thinking": reasoning_text or "",
        "raw": {"content": txt},
    }

def _branch_a(q, choices, gold, model, max_tokens):
    return ask_mcq_once(q, choices, gold, model=model, max_tokens=max_tokens)

# -------------------- branch B --------------------
def ask_mcq_explain(question: str, choices: list[str], gold_letter: str,
                    model: str, max_tokens: int) -> dict:
    letters = letters_for(len(choices))
    sys_prompt, user_prompt = render_mc_prompt_b(question, choices, letters, gold_letter)

    completion = openrouter.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
        extra_body={
            "provider": {"require_parameters": True},
            "response_format": {"type": "json_schema", "json_schema": _schema_explanations_only(letters, gold_letter)},
            "include_reasoning": True,
            "reasoning": {"enabled": True},
        }
    )

    msg = completion.choices[0].message
    txt = msg.content or ""
    reasoning_text = getattr(msg, "reasoning", None) or ""

    try:
        j = json.loads(txt)
    except Exception:
        try:
            j = _coerce_json(txt)
        except Exception:
            logging.warning("JSON parse failed; returning empty object")
            j = {}

    expl_corr = (j.get("explanation_correct") or "").strip()
    expl_inc = j.get("explanations_incorrect") or {}
    if gold_letter in expl_inc:
        expl_inc.pop(gold_letter, None)
    wrong_set = set(L for L in letters if L != gold_letter)
    expl_inc = {k: v for k, v in expl_inc.items() if k in wrong_set}

    return {
        "letters": letters,
        "options": {letters[i]: choices[i] for i in range(len(choices))},
        "gold": gold_letter,
        "explanation_correct": expl_corr,
        "explanations_incorrect": expl_inc,
        "thinking": reasoning_text,
        "raw": {"content": txt},
    }

def _branch_b(q, choices, gold, model, max_tokens):
    return ask_mcq_explain(q, choices, gold, model=model, max_tokens=max_tokens)

# -------------------- branch C --------------------
def ask_mcq_chain(question: str, choices: list[str], gold_letter: str,
                  model: str, max_tokens: int) -> dict:
    letters = letters_for(len(choices))

    sys1, user1 = render_mc_prompt(question, choices, letters)
    comp1 = openrouter.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys1},
            {"role": "user",   "content": user1},
        ],
        max_tokens=max_tokens,
        extra_body={
            "provider": {"require_parameters": True},
            "response_format": {"type": "json_schema", "json_schema": _schema_answer_only(letters)},
            "include_reasoning": True,
            "reasoning": {"enabled": True},
        }
    )
    msg1 = comp1.choices[0].message
    txt1 = msg1.content or ""
    reasoning1 = getattr(msg1, "reasoning", None) or ""

    try:
        j1 = json.loads(txt1)
    except Exception:
        try:
            j1 = _coerce_json(txt1)
        except Exception:
            logging.warning("JSON parse failed (step1); returning empty object")
            j1 = {}

    ans_letter = norm_letter_dyn(j1.get("answer"), letters)
    is_correct = (ans_letter == gold_letter)

    sys2, user2_q = render_mc_prompt_c_review(question, choices, letters, gold_letter)
    assistant_answer_content = txt1 if txt1.strip().startswith("{") else json.dumps({"answer": ans_letter or ""}, ensure_ascii=False)

    comp2 = openrouter.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys2},
            {"role": "user",   "content": user2_q},
            {"role": "assistant", "content": assistant_answer_content},
            {"role": "user", "content": "```Here is your previous reasoning```\n" + (reasoning1 or "")},
            {"role": "user", "content":
                f"Gold answer is {gold_letter}. "
                "Explain briefly why it is correct (explanation_correct), and for each wrong letter explain why it is incorrect (explanations_incorrect). "
                "Do not include the correct letter among explanations_incorrect."
            },
        ],
        max_tokens=max_tokens,
        extra_body={
            "provider": {"require_parameters": True},
            "response_format": {"type": "json_schema", "json_schema": _schema_c_review(letters, gold_letter)},
            "include_reasoning": True,
            "reasoning": {"enabled": True},
        }
    )
    msg2 = comp2.choices[0].message
    txt2 = msg2.content or ""
    reasoning2 = getattr(msg2, "reasoning", None) or ""

    try:
        j2 = json.loads(txt2)
    except Exception:
        try:
            j2 = _coerce_json(txt2)
        except Exception:
            logging.warning("JSON parse failed (step2); returning empty object")
            j2 = {}

    expl_corr = (j2.get("explanation_correct") or "").strip()
    expl_inc = j2.get("explanations_incorrect") or {}
    if gold_letter in expl_inc:
        expl_inc.pop(gold_letter, None)
    wrong_set = set(L for L in letters if L != gold_letter)
    expl_inc = {k: v for k, v in expl_inc.items() if k in wrong_set}

    return {
        "letters": letters,
        "options": {letters[i]: choices[i] for i in range(len(choices))},
        "gold": gold_letter,
        "first_pass": {
            "answer": ans_letter,
            "is_correct": is_correct,
            "thinking": reasoning1,
        },
        "review": {
            "explanation_correct": expl_corr,
            "explanations_incorrect": expl_inc,
            "thinking": reasoning2,
        },
        "raw": {"content_step1": txt1, "content_step2": txt2},
    }

def _branch_c(q, choices, gold, model, max_tokens):
    return ask_mcq_chain(q, choices, gold, model=model, max_tokens=max_tokens)

# -------------------- dataset generation --------------------
def _run_job(job):
    row_id, question, choices, gold_letter, model, max_tokens, branch = job
    try:
        if branch == "A":
            out = _branch_a(question, choices, gold_letter, model=model, max_tokens=max_tokens)
        elif branch == "B":
            out = _branch_b(question, choices, gold_letter, model=model, max_tokens=max_tokens)
        else:
            out = _branch_c(question, choices, gold_letter, model=model, max_tokens=max_tokens)
    except Exception as e:
        logging.warning(f"[idx={row_id}] error: {e}")
        out = {"error": str(e)}

    letters = letters_for(len(choices))
    record_in = {
        "row_id": row_id,
        "question": question,
        "options": {letters[i]: choices[i] for i in range(len(choices))},
        "gold": gold_letter,
        "model": model,
        "branch": branch,
    }
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
):
    assert branch in {"A","B","C"}, "branch must be one of {'A','B','C'}"

    df = pd.read_csv(in_filename, sep="\t", dtype=str, keep_default_na=False)

    total_rows = len(df) if limit is None else min(len(df), int(limit))
    total_chunks = max(1, math.ceil(total_rows / max(1, chunk_size)))

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    written = 0
    stop = False

    with open(out_jsonl, "a", encoding="utf-8") as f, futures.ThreadPoolExecutor(max_workers=chunk_size) as pool:
        for chunk_idx, chunk in tqdm(enumerate(chunker(df, chunk_size)), total=total_chunks, desc=f"Synth {branch}"):
            if stop:
                break

            args_list = []
            for index, row in chunk.iterrows():
                if limit is not None and written >= limit:
                    stop = True
                    break

                q = (row.get("question") or "").strip()
                choices = parse_options(row.get("options") or "[]")
                letters = letters_for(len(choices))
                if len(letters) < 2 or not q:
                    continue

                gold_letter = (
                    norm_letter_dyn(row.get("answer"), letters)
                    or norm_letter_dyn(row.get("answer_index"), letters)
                )
                if not gold_letter:
                    continue

                args_list.append((index, q, choices, gold_letter, model, max_tokens, branch))

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
