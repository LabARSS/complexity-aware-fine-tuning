import os, ast, json, argparse
import pandas as pd
from google import genai
from google.genai import types

from core.prompts.mmlu_branches_aug import *

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

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
    # expected string like: "['optA', 'optB', 'optC', 'optD']"
    try:
        lst = ast.literal_eval(s)
        return list(map(str, lst))
    except Exception:
        # try split; fallback
        s = s.strip().strip("[]")
        parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
        return [p for p in parts if p]

def render_mc_prompt(question, choices, letters):
    opts = "\n".join(f"{letters[i]}) {choices[i]}" for i in range(len(choices)))
    return f"QUESTION:\n{question}\n\nOPTIONS:\n{opts}\n"

def call_json(client, prompt: str) -> dict:
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2,
        thinking_config=types.ThinkingConfig(thinking_budget=-1),  # -1 -- auto choosing (dynamic mode)
        max_output_tokens=1024,
    )
    r = client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
    txt = getattr(r, "text", "") or ""
    # expected JSON; else: trim before first {}-block
    try:
        return json.loads(txt)
    except Exception:
        start, end = txt.find("{"), txt.rfind("}")
        return json.loads(txt[start:end+1]) if start != -1 and end != -1 else {}

def branch_a(client, q, choices, gold):
    letters = letters_for(len(choices))
    base = render_mc_prompt(q, choices, letters)
    allowed = "|".join(letters)
    prompt = (base + p_branch_a(letters=allowed))
    j = call_json(client, prompt)
    ans = norm_letter_dyn(j.get("answer"), letters)
    return {
        "branch": "A",
        "answer": ans,
        "is_correct": ans == gold,
        "rationale": j.get("rationale", ""),
        "key_steps": j.get("key_steps", []),
    }

def branch_b(client, q, choices, gold):
    letters = letters_for(len(choices))
    base = render_mc_prompt(q, choices, letters)
    allowed = "|".join(letters)
    distractor_tpl = "{"+", ".join([f'"{L}":"..."' for L in letters])+"}"
    prompt = (base + p_branch_b(gold, allowed, distractor_tpl))
    j = call_json(client, prompt)
    return {
        "branch": "B",
        "correct_answer": norm_letter_dyn(j.get("correct_answer"), letters),
        "why_correct": j.get("why_correct", ""),
        "distractor_analysis": j.get("distractor_analysis", {}),
    }

def branch_c(client, q, choices, gold):
    letters = letters_for(len(choices))
    base = render_mc_prompt(q, choices, letters)
    allowed = "|".join(letters)
    distractor_tpl = "{"+", ".join([f'"{L}":"..."' for L in letters])+"}"
    # step-1
    prompt1 = (base + p_branch_c_one(allowed))
    j1 = call_json(client, prompt1)
    model_ans = norm_letter_dyn(j1.get("answer"), letters)
    is_correct = model_ans == gold
    # step-2
    prompt2 = (base + p_branch_c_two(model_ans, gold, allowed, distractor_tpl))
    j2 = call_json(client, prompt2)
    return {
        "branch": "C",
        "first_pass": {
            "answer": model_ans,
            "is_correct": is_correct,
            "rationale": j1.get("rationale", ""),
            "key_steps": j1.get("key_steps", []),
        },
        "review": {
            "model_answer": norm_letter_dyn(j2.get("model_answer")),
            "is_correct": bool(j2.get("is_correct")),
            "error_analysis": j2.get("error_analysis"),
            "distractor_analysis": j2.get("distractor_analysis", {}),
        },
    }

def process_tsv(tsv_path, out_jsonl, limit=None):
    client = genai.Client()
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)

    with open(out_jsonl, "a", encoding="utf-8") as f:
        for i, row in df.iterrows():
            if limit is not None and i >= limit: break

            question = row.get("question", "").strip()
            choices = parse_options(row.get("options", "[]"))
            letters = letters_for(len(choices))
            gold = norm_letter_dyn(row.get("answer", ""), letters) or norm_letter_dyn(row.get("answer_index", ""), letters)

            if len(choices) < 2 or gold not in letters or not question:
                continue  # bad strings
            meta = {
                "src": row.get("src", ""),
                "category": row.get("category", ""),
                "question_id": row.get("question_id", ""),
                "meta_cluster": row.get("meta_cluster", ""),
                "base_cluster": row.get("base_cluster", ""),
                "total_tokens": row.get("total_tokens", ""),
            }

            record_in = {
                "subject": meta.get("category") or meta.get("src"),
                "question": question,
                "options": {letters[i]: choices[i] for i in range(len(choices))},
                "gold": gold,
                "model": MODEL,
                "meta": meta,
            }

            outA = branch_a(client, question, choices, gold)
            outB = branch_b(client, question, choices, gold)
            outC = branch_c(client, question, choices, gold)

            for out in (outA, outB, outC):
                f.write(json.dumps({"input": record_in, "output": out}, ensure_ascii=False) + "\n")

            print(f"[{i+1}/{len(df)}] done", end="\r")

def main():
    ap = argparse.ArgumentParser(description="Generate MMLU-style synthetic data from TSV.")
    ap.add_argument("--tsv", required=True, help="Path to input .tsv (e.g., mmlu_pro_stem.tsv)")
    ap.add_argument("--out", default="work/complexity-aware-sft/complexity-aware-fine-tuning/data/out/distillation/synth-aug-mmlu.jsonl", help="Output JSONL path")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of rows (optional)")
    args = ap.parse_args()

    process_tsv(args.tsv, args.out, limit=args.limit)
    print(f"\nSaved to {args.out}")

if __name__ == "__main__":
    main()
