import torch

def vector_sum(a: torch.Tensor):
    """Sum elements of a via a dot product with a ones vector (no torch.sum)."""
    return torch.matmul(a,torch.ones(len(a)))
    