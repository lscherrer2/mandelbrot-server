import torch


def determine_device():
    """Determine the best available device: cuda, mps, or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
