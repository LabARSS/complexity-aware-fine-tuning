from typing import Any, Optional, Union

import torch
from torch import nn
from transformers.trainer_seq2seq import Seq2SeqTrainer


class CoTEvalTrainer(Seq2SeqTrainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        cot_labels = None
        if "cot_labels" in inputs:
            cot_labels = inputs.pop("cot_labels")

        self.args.predict_with_generate = True
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
        )
        self.args.predict_with_generate = False

        return None, generated_tokens, cot_labels or labels
