def p_branch_a(letters):
    p = f"""
            Return JSON only with:
            {{"answer":"{letters}","rationale":"1-3 sentences (concise)","key_steps":["fact1","fact2","fact3"]}}

            Answer the MCQ briefly and factually (no step-by-step reasoning).
            """
    return p

def p_branch_b(gold, letters, distractor_tpl):
    p = f"""
        The correct answer is "{gold}".
        Return JSON only:
        {{"correct_answer":"{letters}",
        "why_correct":"1-3 sentences",
        "distractor_analysis": {distractor_tpl} }}
        Explain why the correct is right and others are wrong (concise, verifiable).
        """ 
    return p

def p_branch_c_one(allowed):
    p = f"""
        Return JSON only:
        {{"answer":"{allowed}","rationale":"1-2 sentences","key_steps":["fact1","fact2"]}}
        """
    return p

def p_branch_c_two(model_ans, gold, allowed, distractor_tpl):
    p = f"""
    Your previous answer: "{model_ans}".
    Gold (correct) answer: "{gold}".
    Return JSON only:
    {{"model_answer":"{allowed}",
    "is_correct": true/false,
    "error_analysis": "1-2 sentences or null",
    "distractor_analysis": {distractor_tpl} }}
    Keep it concise; no step-by-step reasoning.
    """
    return p
