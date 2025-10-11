from dataclasses import dataclass, field
from typing import Any

import torch

from core.complexity_estimation.entropy.logit_entropy import compute_entropy_from_logits


@dataclass
class LogitSeqStats:
    # Generated token selected greedily (no randomness, next token - the most likely one)
    greedy_tokens: list[torch.Tensor] = field(default_factory=list)
    # List of entropies for every generated token
    entropies: list[float] = field(default_factory=list)
    # List of raw probabilities for logits with non-zero probabilities for every generated token
    every_token_stats: list[list[dict[str, Any]]] = field(default_factory=list)


def collect_logit_sequence_stats(logits: list[torch.Tensor], in_batch_idx=0):
    """
    Parameters:
    ----------
    logits : torch.Tensor
        Logits for the entire generated sequence. Assumes batch size of 1.
        Pass here "scores" from "model.generate".
        Dim: list[tensor(1 x dictionary_size)]
    """
    stats = LogitSeqStats()
    for i in range(len(logits)):
        # generated token position, batch_dim
        token_logits = logits[i][in_batch_idx]
        token_entropy = compute_entropy_from_logits(token_logits)
        stats.entropies.append(token_entropy)

        probabilities = torch.softmax(token_logits, dim=-1)
        # Set small cut-off value
        mask = probabilities > 1e-5
        nonzero_prob_indices = torch.nonzero(mask)
        nonzero_probs = probabilities[nonzero_prob_indices]
        idx_prob_pairs_list = list(zip(nonzero_prob_indices.cpu().numpy(), nonzero_probs.cpu().numpy()))
        position_result = [
            {
                "token_idx": pair[0].item(),
                "token_prob": pair[1].item(),
            }
            for pair in idx_prob_pairs_list
        ]
        stats.every_token_stats.append(position_result)

        greedy_token = token_logits.argmax(dim=-1)
        stats.greedy_tokens.append(greedy_token)

    return stats
