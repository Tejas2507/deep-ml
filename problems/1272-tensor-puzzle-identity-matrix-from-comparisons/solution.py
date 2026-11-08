import torch

def eye(n: int) -> torch.Tensor:
    """n x n identity via comparisons (no torch.eye)."""
    i = torch.arange(n)
    mask = i[:,None] == i[None,:]
    return mask.to(torch.float64)

    
