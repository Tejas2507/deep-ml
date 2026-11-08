import numpy as np

def diag(A: np.ndarray) -> np.ndarray:
    """Return the main diagonal of square matrix A."""
    print(torch.arange(A.shape[0]), torch.arange(A.shape[0]))


    return np.asarray([A[i][i] for i in range(len(A))])
