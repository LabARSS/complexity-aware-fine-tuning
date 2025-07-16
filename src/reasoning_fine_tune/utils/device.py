import torch

DEVICE = torch.device("cpu")
DEVICE_MAP = "cpu"
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEVICE_MAP = "auto"
if torch.mps.is_available():
    DEVICE = torch.device("mps")
    DEVICE_MAP = "mps"


def move_batch_to_device(batch, device):
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
