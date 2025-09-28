# template prompts for synth-aug-mmlu.py (exp.: explain-mmlu.py)


def p_json_guardrails():
    return """
You MUST respond with valid JSON ONLY. Do not include any prose, code fences, or explanations outside JSON.
If additional context is needed, include it ONLY inside the JSON fields.
"""

def p_branch_a(letters):
    return f"""
Return JSON ONLY with the following schema:
{{
  "answer": "{letters}",
  "rationale": "concise 1-2 sentence justification (no fluff)",
  "key_steps": ["step1","step2","step3"],
  "thinking": "step-by-step reasoning tokens leading to the answer"
}}
Ensure "answer" is one of {letters}. Keep rationale concise but factual.
"""

def p_branch_b(gold, letters, distractor_tpl):
    return f"""
        The correct answer is "{gold}".
        Return JSON only:
        {{"correct_answer":"{letters}",
        "why_correct": "step-by-step reasoning showing why the gold option is correct",
        "distractor_analysis": {distractor_tpl} }}
        Each entry in "distractor_analysis" must explain why that option is incorrect.
        """ 

def p_branch_c_one(allowed):
    return f"""
        Return JSON only:
        {{"answer":"{allowed}","rationale":"short justification (1-2 sentences)","key_steps":["step1","step2"], "thinking": "step-by-step reasoning tokens used to pick the answer"}}
        """

def p_branch_c_two(model_ans, gold, allowed, distractor_tpl):
    return f"""
    Your previous answer: "{model_ans}".
    Gold (correct) answer: "{gold}".
    Return JSON only:
    {{"model_answer":"{allowed}",
    "is_correct": true/false,
    "error_analysis": "step-by-step reasoning of the mistake (if any) or null",
    "distractor_analysis": {distractor_tpl} }}
    """
