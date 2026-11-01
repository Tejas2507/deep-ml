
import numpy as np

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.

    Returns:
        L: Lower triangular matrix as a 2D list, or -1 if decomposition fails.
    """
    A = np.array(A, dtype=float)

    # Check that A is square
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return -1

    # Check that A is symmetric
    if not np.allclose(A, A.T):
        return -1

    n = A.shape[0]
    L = np.zeros((n, n))

    try:
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i, k] * L[j, k] for k in range(j))

                if i == j:
                    value = A[i, i] - s

                    # Diagonal must be strictly positive
                    if value <= 0:
                        return -1

                    L[i, j] = np.sqrt(value)
                else:
                    L[i, j] = (A[i, j] - s) / L[j, j]

    except (ValueError, ZeroDivisionError, FloatingPointError):
        return -1

    return L.tolist()