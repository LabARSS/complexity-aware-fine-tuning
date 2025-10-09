import os, ast, json
from concurrent import futures

import pandas as pd
from tqdm import tqdm

from core.utils.openrouter import openrouter
from core.utils.chunker import chunker

from core.prompts.mmlu_branches_aug import *

# defaults
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-0528")
DEFAULT_MAX_TOKENS = int(os.getenv("OPENROUTER_MAX_TOKENS", "2048"))
CHUNK_SIZE = int(os.getenv("SYNTH_CHUNK_SIZE", "2"))
DUMP_EVERY = int(os.getenv("SYNTH_DUMP_EVERY", "10"))
DEFAULT_BRANCHES = tuple((os.getenv("SYNTH_BRANCHES") or "B,C").split(","))

ALL_LETTERS = [chr(c) for c in range(ord("A"), ord("Z")+1)]

def letters_for(n: int):
    n = max(0, min(int(n), 26))
    return ALL_LETTERS[:n]

def norm_letter_dyn(x, letters):
    s = ("" if x is None else str(x)).strip().upper()
    if s in letters: return s
    if s.isdigit():
        i = int(s)
        if 0 <= i < len(letters): return letters[i]
    return ""

def parse_options(s):
    try:
        lst = ast.literal_eval(s)
        return list(map(str, lst))
    except Exception:
        s = s.strip().strip("[]")
        parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
        return [p for p in parts if p]

def render_mc_prompt(question, choices, letters):
    opts = "\n".join(f"{letters[i]}) {choices[i]}" for i in range(len(choices)))
    return f"QUESTION:\n{question}\n\nOPTIONS:\n{opts}\n"

def _openrouter_json(sys_prompt: str, user_prompt: str, model: str, max_tokens: int) -> dict:

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    completion = openrouter.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,

        # check https://openrouter.ai/docs/use-cases/reasoning-tokens#enable-reasoning-with-default-config
        extra_body={"include_reasoning": True}
        # reasoning={"enabled": True},
        # response_format={"type": "json_object"},
    )
    txt = completion.choices[0].message.content or ""
    reasoning_text = getattr(completion.choices[0].message, "reasoning", None)

    j = None
    try:
        j = json.loads(txt)
    except Exception:
        start, end = txt.find("{"), txt.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                j = json.loads(txt[start:end+1])
            except Exception:
                pass
    j = j or {}
    if reasoning_text and "thinking" not in j:
        j["thinking"] = reasoning_text
    return j

def branch_a(q, choices, gold, model, max_tokens):
    letters = letters_for(len(choices))
    allowed = "|".join(letters)
    base = render_mc_prompt(q, choices, letters)
    prompt = (p_json_guardrails() + "\n" + base + p_branch_a(letters=allowed))
    j = _openrouter_json("You return STRICT JSON.", prompt, model, max_tokens)
    ans = norm_letter_dyn(j.get("answer"), letters)
    return {
        "branch": "A",
        "answer": ans,
        "is_correct": ans == gold,
        "rationale": j.get("rationale", ""),
        "key_steps": j.get("key_steps", []),
        "thinking": j.get("thinking", ""),
    }

def branch_b(q, choices, gold, model, max_tokens):
    letters = letters_for(len(choices))
    allowed = "|".join(letters)
    base = render_mc_prompt(q, choices, letters)
    distractor_tpl = "{" + ", ".join([f'"{L}":"..."' for L in letters if L != gold]) + "}"
    prompt = (p_json_guardrails() + "\n" + base + p_branch_b(gold, allowed, distractor_tpl))
    j = _openrouter_json("You return STRICT JSON.", prompt, model, max_tokens)
    return {
        "branch": "B",
        "correct_answer": norm_letter_dyn(j.get("correct_answer"), letters),
        "why_correct": j.get("why_correct", ""),
        "distractor_analysis": j.get("distractor_analysis", {}),
    }

def branch_c(q, choices, gold, model, max_tokens):
    letters = letters_for(len(choices))
    allowed = "|".join(letters)
    base = render_mc_prompt(q, choices, letters)
    distractor_tpl = "{" + ", ".join([f'"{L}":"..."' for L in letters if L != gold]) + "}"

    prompt1 = (p_json_guardrails() + "\n" + base + p_branch_c_one(allowed))
    j1 = _openrouter_json("You return STRICT JSON.", user_prompt=prompt1, model=model, max_tokens=max_tokens)
    model_ans = norm_letter_dyn(j1.get("answer"), letters)
    is_correct = model_ans == gold

    prompt2 = (p_json_guardrails() + "\n" + base + p_branch_c_two(model_ans, gold, allowed, distractor_tpl))
    j2 = _openrouter_json("You return STRICT JSON.", user_prompt=prompt2, model=model, max_tokens=max_tokens)

    return {
        "branch": "C",
        "first_pass": {
            "answer": model_ans,
            "is_correct": is_correct,
            "rationale": j1.get("rationale", ""),
            "key_steps": j1.get("key_steps", []),
            "thinking": j1.get("thinking", ""),
        },
        "review": {
            "model_answer": norm_letter_dyn(j2.get("model_answer"), letters),
            "is_correct": bool(j2.get("is_correct")),
            "error_analysis": j2.get("error_analysis"),
            "distractor_analysis": j2.get("distractor_analysis", {}),
        },
    }

def _build_record_in(row_dict, question, choices, letters, gold, model):
    subject = row_dict.get("category") or row_dict.get("src") or ""
    meta = {k: v for k, v in row_dict.items() if k not in ("question","options","answer","answer_index")}
    return {
        "subject": subject,
        "question": question,
        "options": {letters[i]: choices[i] for i in range(len(choices))},
        "gold": gold,
        "model": model,
        "meta": meta,
    }

def _prepare_jobs(df, model, max_tokens, limit, branches):
    jobs = []
    for i, row in enumerate(df.itertuples(index=True)):
        if limit is not None and i >= limit:
            break
        row_dict = df.iloc[row.Index].to_dict()
        question = (row_dict.get("question") or "").strip()
        choices = parse_options(row_dict.get("options") or "[]")
        letters = letters_for(len(choices))
        gold = norm_letter_dyn(row_dict.get("answer"), letters) or norm_letter_dyn(row_dict.get("answer_index"), letters)
        if len(choices) < 2 or gold not in letters or not question:
            continue
        record_in = _build_record_in(row_dict, question, choices, letters, gold, model)
        if "A" in branches:
            jobs.append((row.Index, "A", {"question": question, "choices": choices, "gold": gold, "record_in": record_in, "letters": letters}))
        if "B" in branches:
            jobs.append((row.Index, "B", {"question": question, "choices": choices, "gold": gold, "record_in": record_in, "letters": letters}))
        if "C" in branches:
            jobs.append((row.Index, "C", {"question": question, "choices": choices, "gold": gold, "record_in": record_in, "letters": letters}))
    return jobs

def _run_job(args):
    (index, branch_id, payload, model, max_tokens) = args
    q = payload["question"]; ch = payload["choices"]; gold = payload["gold"]
    if branch_id == "A":
        out = branch_a(q, ch, gold, model, max_tokens)
    elif branch_id == "B":
        out = branch_b(q, ch, gold, model, max_tokens)
    else:
        out = branch_c(q, ch, gold, model, max_tokens)
    return index, branch_id, payload["record_in"], out

def synth_on_dataset(
    in_filename: str,
    out_jsonl: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    dump_every: int = DUMP_EVERY,
    limit: int | None = None,
    branches: tuple[str, ...] = DEFAULT_BRANCHES
):
    """Generate synthetic A/B/C branches from TSV and write JSONL lines."""
    df = pd.read_csv(in_filename, sep="\t", dtype=str, keep_default_na=False)
    branches = tuple(b for b in (map(str.strip, branches)) if b in {"A","B","C"})
    if not branches:
        branches = ("B","C")
    jobs = _prepare_jobs(df, model, max_tokens, limit, branches)
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    f = open(out_jsonl, "a", encoding="utf-8")
    with futures.ThreadPoolExecutor(max_workers=CHUNK_SIZE) as pool:
        args_list = [(index, branch_id, payload, model, max_tokens) for (index, branch_id, payload) in jobs]

        total_batches = (len(args_list) + CHUNK_SIZE - 1) // CHUNK_SIZE
        for k in tqdm(range(0, len(args_list), CHUNK_SIZE), total=total_batches, desc="Synthesizing"):
            batch = args_list[k : k + CHUNK_SIZE]
            results = list(pool.map(_run_job, batch))

            for index, branch_id, record_in, out in results:
                f.write(json.dumps({"input": record_in, "output": out}, ensure_ascii=False) + "\n")

            if (k // CHUNK_SIZE) % dump_every == 0:
                f.flush()

    f.close()
    print(f"Saved to {out_jsonl}. Total inputs: {df.shape[0]}; outputs: {len(jobs)}.")
    return out_jsonl
