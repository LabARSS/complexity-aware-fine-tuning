import os, ast, json, re, logging
from concurrent import futures
import pandas as pd

from core.utils.openrouter import openrouter

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

ALL_LETTERS = [chr(c) for c in range(ord("A"), ord("Z")+1)]
chunk_size = 30

def letters_for(n: int):
    n = max(0, min(int(n), 26))
    return ALL_LETTERS[:n]

def parse_options(s):
    try:
        lst = ast.literal_eval(s)
        return list(map(str, lst))
    except Exception:
        s = (s or "").strip().strip("[]")
        parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
        return [p for p in parts if p]

def norm_letter_dyn(x, letters):
    s = ("" if x is None else str(x)).strip().upper()
    if s in letters:
        return s
    if s.isdigit():
        i = int(s)
        if 0 <= i - 1 < len(letters): return letters[i - 1]
        if 0 <= i < len(letters):     return letters[i]
    return ""

def render_mc_prompt(question, choices, letters):
    opts = "\n".join(f"{letters[i]}) {choices[i]}" for i in range(len(choices)))
    sys_prompt = (
        "Return STRICT JSON ONLY as {\"answer\": \"<LETTER>\"} where <LETTER> "
        f"is one uppercase letter from {letters}. No extra fields, no code fences."
    )
    user_prompt = f"QUESTION:\n{question}\n\nOPTIONS:\n{opts}\n"
    return sys_prompt, user_prompt

def render_mc_prompt_c_review(question, choices, letters, gold_letter):
    opts = "\n".join(f"{letters[i]}) {choices[i]}" for i in range(len(choices)))
    wrong_letters = [L for L in letters if L != gold_letter]
    wrong_list = ", ".join(wrong_letters)
    sys_prompt = (
        "Return STRICT JSON ONLY as {\"explanation_correct\":\"...\","
        "\"explanations_incorrect\": {\"<WRONG_LETTER>\": \"...\", ... }}. "
        "Do not include Markdown or code fences. "
        f"Use only the wrong option letters: {wrong_list} as keys in explanations_incorrect."
    )
    user_prompt = (
        f"QUESTION:\n{question}\n\nOPTIONS:\n{opts}\n"
    )
    return sys_prompt, user_prompt

def _schema_answer_only(letters):
    return {
        "name": "mcq_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": { "answer": {"type": "string", "enum": letters} },
            "required": ["answer"],
            "additionalProperties": False
        }
    }

def _schema_c_review(letters, gold_letter):
    wrong = [L for L in letters if L != gold_letter]
    return {
        "name": "mcq_branch_c_review",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "explanation_correct": {"type": "string"},
                "explanations_incorrect": {
                    "type": "object",
                    "properties": { k: {"type": "string"} for k in wrong },
                    "required": wrong,
                    "additionalProperties": False
                }
            },
            "required": ["explanation_correct", "explanations_incorrect"],
            "additionalProperties": False
        }
    }

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

def ask_mcq_chain(question: str, choices: list[str], gold_letter: str,
                  model: str, max_tokens: int) -> dict:
    letters = letters_for(len(choices))

    # Step 1: answer + reasoning
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

    # Step 2: review + reasoning; include prior assistant content and prior reasoning
    sys2, user2_q = render_mc_prompt_c_review(question, choices, letters, gold_letter)

    assistant_answer_content = txt1 if txt1.strip().startswith("{") else json.dumps({"answer": ans_letter or ""}, ensure_ascii=False)

    comp2 = openrouter.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys2},
            {"role": "user",   "content": user2_q},
            {"role": "assistant", "content": assistant_answer_content},
            {"role": "user", "content": "```Вот твой предыдущий reasoning```\n" + (reasoning1 or "")},
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

def _run_job(job):
    row_id, question, choices, gold_letter, model, max_tokens = job
    try:
        out = ask_mcq_chain(question, choices, gold_letter, model=model, max_tokens=max_tokens)
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
    }
    return row_id, record_in, out

def _iter_chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def synth_on_dataset(
    in_filename: str,
    out_jsonl: str,
    model: str,
    max_tokens: int,
    dump_every: int,
    limit: int | None = None
):
    df = pd.read_csv(in_filename, sep="\t", dtype=str, keep_default_na=False)

    jobs = []
    for row in df.itertuples():
        i = row.Index
        if limit is not None and i >= limit:
            break

        q = (df.at[i, "question"] or "").strip()
        choices = parse_options(df.at[i, "options"] or "[]")
        letters = letters_for(len(choices))
        if len(letters) < 2 or not q:
            continue

        gold_letter = (
            norm_letter_dyn(df.at[i, "answer"] if "answer" in df.columns else None, letters)
            or norm_letter_dyn(df.at[i, "answer_index"] if "answer_index" in df.columns else None, letters)
        )
        if not gold_letter:
            continue

        jobs.append((i, q, choices, gold_letter, model, max_tokens))

    if not jobs:
        logging.info("No valid jobs to run.")
        return out_jsonl

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    written = 0
    with open(out_jsonl, "a", encoding="utf-8") as f, futures.ThreadPoolExecutor(max_workers=chunk_size) as pool:
        for batch in _iter_chunks(jobs, chunk_size):
            for row_id, record_in, record_out in pool.map(_run_job, batch):
                f.write(json.dumps({"input": record_in, "output": record_out}, ensure_ascii=False) + "\n")
                written += 1
                if dump_every > 0 and (written % dump_every == 0):
                    f.flush()

    print(f"Saved to {out_jsonl}. Total inputs: {df.shape[0]}; jobs run: {len(jobs)}; written: {written}.")
    return out_jsonl
