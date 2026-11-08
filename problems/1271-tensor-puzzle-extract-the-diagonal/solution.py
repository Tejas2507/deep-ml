import torch
def diag(A: torch.Tensor) -> torch.Tensor:
    return A[torch.arange(A.shape[0]), torch.arange(A.shape[0])]