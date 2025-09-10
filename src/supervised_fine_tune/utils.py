import re
from typing import Optional
import torch

ANSWER_MARKER = ["[[", "]]" ]
ANSWER_PATTERN = re.compile(r'.*?\[\[\s*(\d+)\s*\]\].*?', re.DOTALL)


def cot_sys_prompt(use_cot: bool, subject: Optional[str] = None) -> str:
    if subject:
        base = f"The following are multiple choice questions about {subject}."
    else:
        base = "The following are multiple choice questions."
    if use_cot:
        base += f" Explain your thinking step-by-step. At the end, write the correct answer number using {ANSWER_MARKER[0]}number{ANSWER_MARKER[1]} format."
    else:
        base += f" Analyze question and answer with only one number in {ANSWER_MARKER[0]}number{ANSWER_MARKER[1]} format."
    return base


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor, tokenizer) -> torch.Tensor:
    predictions = torch.argmax(logits, dim=-1)
    shifted_labels = labels[..., 1:].contiguous()
    shifted_predictions = predictions[..., :-1].contiguous()
    mask = shifted_labels != -100
    mask = mask & (shifted_labels != tokenizer.eos_token_id)
    correct = (shifted_predictions == shifted_labels) & mask
    # protect division
    denom = mask.float().sum()
    return correct.float().sum() / (denom if denom > 0 else 1.0)