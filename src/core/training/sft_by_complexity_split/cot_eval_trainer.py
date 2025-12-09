import csv
import os
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer_utils import EvalLoopOutput


class CoTEvalTrainer(Seq2SeqTrainer):
    def __init__(self, invalid_answers_save_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.invalid_answers_save_path = invalid_answers_save_path
        # Only remove on main process to avoid race conditions
        if self.is_world_process_zero() and os.path.exists(self.invalid_answers_save_path):
            os.remove(self.invalid_answers_save_path)
        self.file_initialized = False

        self.skip_eval_datasets = {}

        # Wrap compute_metrics to run only on main process
        if self.compute_metrics is not None:
            original_compute_metrics = self.compute_metrics

            def compute_metrics_wrapper(eval_pred, compute_result):
                # Only compute metrics on main process to save resources
                if self.is_world_process_zero():
                    return original_compute_metrics(eval_pred, compute_result)
                else:
                    # Return empty dict on other processes
                    return {}

            self.compute_metrics = compute_metrics_wrapper

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Check if this is CoT evaluation (all samples in batch have same mode)
        is_cot_eval = inputs.get("cot", torch.tensor([False]))[0].item()

        if is_cot_eval:
            self.args.predict_with_generate = True

        # Remove question_id and cot from inputs before passing to model
        # They will still be available in eval_pred.inputs for metrics computation
        # because the Trainer gathers them via include_for_metrics=["inputs"]
        inputs_filtered = {k: v for k, v in inputs.items() if k not in ["question_id", "cot"]}

        loss, generated_tokens, labels = super().prediction_step(
            model, inputs_filtered, prediction_loss_only, ignore_keys, **gen_kwargs
        )

        self.args.predict_with_generate = False

        return None if is_cot_eval else loss, generated_tokens, labels

    def evaluation_loop(self, *args, **kwargs) -> EvalLoopOutput:
        metric_key_prefix = kwargs["metric_key_prefix"]
        epoch = self.state.epoch
        prefixed_accuracy = f"{metric_key_prefix}_accuracy"
        prefixed_incorrect_answers = f"{metric_key_prefix}_incorrect_answers"

        if metric_key_prefix in self.skip_eval_datasets:
            last_epoch = self.skip_eval_datasets[metric_key_prefix]
            print(f"Skipping eval on {metric_key_prefix} at epoch {epoch}! Last executed epoch = {last_epoch}.")
            return EvalLoopOutput(
                predictions=np.array([]), label_ids=None, metrics={prefixed_accuracy: 0}, num_samples=0
            )

        eval_loop_output = super().evaluation_loop(*args, **kwargs)

        assert eval_loop_output.metrics is not None

        # Only save incorrect answers on the main process to avoid duplicates
        if self.is_world_process_zero():
            assert prefixed_accuracy in eval_loop_output.metrics
            assert prefixed_incorrect_answers in eval_loop_output.metrics

            incorrect_answers = eval_loop_output.metrics.pop(prefixed_incorrect_answers)
            assert isinstance(incorrect_answers, list)

            for item in incorrect_answers:
                item["dataset"] = metric_key_prefix
                item["epoch"] = epoch

            new_incorrect_answers_df = pd.DataFrame(incorrect_answers)
            new_incorrect_answers_df.to_csv(
                self.invalid_answers_save_path,
                sep="\t",
                mode="a",
                header=not self.file_initialized,
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
            self.file_initialized = True

            if eval_loop_output.metrics[prefixed_accuracy] == 0:
                self.skip_eval_datasets[metric_key_prefix] = epoch

        return eval_loop_output
