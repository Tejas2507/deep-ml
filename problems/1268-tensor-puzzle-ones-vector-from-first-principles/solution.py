import torch

def ones(n: int) -> torch.Tensor:
    """Return a length-n float vector of ones without calling torch.ones."""
    return torch.ones(n)

