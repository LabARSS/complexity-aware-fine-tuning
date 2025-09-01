from typing import Callable, Dict, List, Optional
from transformers import PreTrainedTokenizerBase

from core.prompts.mmlu_cot_answer import answer_marker as COT_MARKERS

def make_prompt_builder(
    get_sys_prompt: Callable[[dict], str],
    get_user_prompt: Callable[[str, str], str],
    get_subject_from_row: Callable[[dict], str],
    get_question_from_row: Callable[[dict], str],
    get_options_from_row: Callable[[dict], str],
) -> Callable[[dict], List[Dict[str, str]]]:
    """Билдер промптов для схем CoT/Single-token"""
    def _builder(row) -> List[Dict[str, str]]:
        sys_prompt = get_sys_prompt(get_subject_from_row(row))
        user_prompt = get_user_prompt(get_question_from_row(row), get_options_from_row(row))
        return [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}]
    return _builder

def single_token_answer_extractor(gen_ids: List[int], gen_text: str, tokenizer: PreTrainedTokenizerBase):
    return {"answer": gen_text.strip()}

def cot_answer_extractor(gen_ids: List[int], gen_text: str, tokenizer: PreTrainedTokenizerBase):
    """
    Находим answer по текстовым маркерам, как было в estimate_cot_entropy.py:
    ищем [answer_marker_start, answer_marker_end] по строке,
    считаем индекс токена ответа как start+1.
    """
    start_m, end_m = COT_MARKERS
    out = {"answer": "", "ans_token_index": None, "think_text": None}
    acc = ""
    start_i = -1
    end_i = -1
    for i, tid in enumerate(gen_ids):
        token_str = tokenizer.decode([tid], skip_special_tokens=True)
        acc += token_str
        if start_i == -1 and start_m in acc:
            start_i = i
        elif start_i != -1 and end_i == -1 and end_m in acc:
            end_i = i
            break

    if start_i != -1 and end_i != -1:
        ans_idx = start_i + 1
        out["ans_token_index"] = ans_idx
        out["answer"] = tokenizer.decode([gen_ids[ans_idx]], skip_special_tokens=True).strip()
        if start_i > 0:
            out["think_text"] = tokenizer.decode(gen_ids[:start_i], skip_special_tokens=True).strip()
    return out
