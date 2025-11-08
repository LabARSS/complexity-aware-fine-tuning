ALL_LETTERS = [chr(c) for c in range(ord("A"), ord("Z")+1)]

def render_mc_prompt(question, choices, letters):
    opts = "\n".join(f"{letters[i]}) {choices[i]}" for i in range(len(choices)))
    sys_prompt = (
        'Return STRICT JSON ONLY as {"answer": "<LETTER>"} where <LETTER> '
        f'is one uppercase letter from {letters}. No extra fields, no code fences.'
    )
    user_prompt = f"QUESTION:\n{question}\n\nOPTIONS:\n{opts}\n"
    return sys_prompt, user_prompt

def render_mc_prompt_b(question, choices, letters, gold_letter):
    opts = "\n".join(f"{letters[i]}) {choices[i]}" for i in range(len(choices)))
    wrong_letters = [L for L in letters if L != gold_letter]
    wrong_list = ", ".join(wrong_letters)
    sys_prompt = (
        'Return STRICT JSON ONLY as {"explanation_correct":"...",'
        '"explanations_incorrect": {"<WRONG_LETTER>": "...", ... }}. '
        "Do not include Markdown or code fences. "
        f"Use only the wrong option letters: {wrong_list} as keys in explanations_incorrect."
    )
    user_prompt = (
        f"QUESTION:\n{question}\n\nOPTIONS:\n{opts}\n\n"
        f"CORRECT ANSWER: {gold_letter}\n"
        "Explain concisely why the correct option is correct (explanation_correct), "
        "and for each wrong option letter explain why it is incorrect (explanations_incorrect)."
    )
    return sys_prompt, user_prompt

def render_mc_prompt_c_review(question, choices, letters, gold_letter):
    opts = "\n".join(f"{letters[i]}) {choices[i]}" for i in range(len(choices)))
    wrong_letters = [L for L in letters if L != gold_letter]
    wrong_list = ", ".join(wrong_letters)
    sys_prompt = (
        'Return STRICT JSON ONLY as {"explanation_correct":"...",'
        '"explanations_incorrect": {"<WRONG_LETTER>": "...", ... }}. '
        "Do not include Markdown or code fences. "
        f"Use only the wrong option letters: {wrong_list} as keys in explanations_incorrect."
    )
    user_prompt = f"QUESTION:\n{question}\n\nOPTIONS:\n{opts}\n"
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

def _schema_explanations_only(letters, gold_letter):
    wrong = [L for L in letters if L != gold_letter]
    return {
        "name": "mcq_branch_b",
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
