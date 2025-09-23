def p_branch_a():
    p = """
            Return JSON only with:
            {"answer":"A|B|C|D","rationale":"1-3 sentences (concise)","key_steps":["fact1","fact2","fact3"]}

            Answer the MCQ briefly and factually (no step-by-step reasoning).
            """
    return p

def p_branch_b(gold):
    p = f"""
        The correct answer is "{gold}".
        Return JSON only:
        {{"correct_answer":"A|B|C|D",
        "why_correct":"1-3 sentences",
        "distractor_analysis": {{"A":"...", "B":"...", "C":"...", "D":"..."}} }}
        Explain why the correct is right and others are wrong (concise, verifiable).
        """ 
    return p

def p_branch_c_one():
    p = """
        Return JSON only:
        {"answer":"A|B|C|D","rationale":"1-2 sentences","key_steps":["fact1","fact2"]}
        """
    return p

def p_branch_c_two(model_ans, gold):
    p = f"""
    Your previous answer: "{model_ans}".
    Gold (correct) answer: "{gold}".
    Return JSON only:
    {{"model_answer":"A|B|C|D",
    "is_correct": true/false,
    "error_analysis": "1-2 sentences or null",
    "distractor_analysis": {{"A":"...", "B":"...", "C":"...", "D":"..."}} }}
    Keep it concise; no step-by-step reasoning.
    """
    return p
