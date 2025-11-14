from pathlib import Path
from typing import Optional

import json
import torch
from transformers import TrainerCallback, TrainerState, TrainerControl

try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

class SaveOnEpochEndAndLogWeightsCallback(TrainerCallback):
    def __init__(self, output_dir: str, save_full_model_for_non_lora: bool = False):
        self.output_dir = Path(output_dir)
        self.weights_dir = self.output_dir / "weights_by_epoch"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.save_full_model_for_non_lora = save_full_model_for_non_lora

    def _is_lora_model(self, model) -> bool:
        return _HAS_PEFT and isinstance(model, PeftModel)

    def _save_json_marker(self, epoch_value: float):
        # mini marker for progress
        marker = {
            "epoch": epoch_value,
        }
        with open(self.output_dir / "last_completed_epoch.json", "w", encoding="utf-8") as f:
            json.dump(marker, f)

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Force savings
        control.should_save = True

        # 2) Log weights
        model = kwargs.get("model", None)
        if model is not None:
            epoch_int = int(state.epoch) if state.epoch is not None else -1
            tag = f"epoch_{epoch_int:02d}"

            try:
                if self._is_lora_model(model):
                    # Only adapters
                    epoch_dir = self.weights_dir / tag
                    epoch_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(epoch_dir), safe_serialization=True)
                elif self.save_full_model_for_non_lora:
                    # Can be big files for full weights, be careful!
                    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    torch.save(sd, self.weights_dir / f"{tag}_full_model.pt")
            except Exception as e:
                print(f"[WARN] Failed to dump weights for {tag}: {e}")

        # Mark of the end of epoch
        try:
            self._save_json_marker(state.epoch if state.epoch is not None else -1)
        except Exception as e:
            print(f"[WARN] Failed to write last_completed_epoch.json: {e}")

        return control
