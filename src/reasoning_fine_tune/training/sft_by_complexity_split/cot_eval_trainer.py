from typing import Any, Optional, Union

import torch
from torch import nn
from transformers.trainer_seq2seq import Seq2SeqTrainer


class CoTEvalTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_cot_eval = False
        compute_metrics = self.compute_metrics

        def compute_metrics_enhanced(eval_pred, compute_result):
            return compute_metrics(eval_pred, compute_result, self.is_cot_eval)

        self.compute_metrics = compute_metrics_enhanced

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        self.is_cot_eval = False
        if "cot" in inputs:
            self.is_cot_eval = True
            inputs.pop("cot")
            self.args.predict_with_generate = True

        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
        )

        self.args.predict_with_generate = False

        return None if self.is_cot_eval else loss, generated_tokens, labels
