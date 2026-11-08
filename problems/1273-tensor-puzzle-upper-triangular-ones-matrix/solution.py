import numpy as np

def triu_ones(n: int) -> np.ndarray:
    """n x n upper-triangular matrix of ones (including diagonal)."""
    i = np.arange(n)
    mask = i[:,None] <= i[None,:]
    return mask.astype(float)

