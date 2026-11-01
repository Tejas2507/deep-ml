import numpy as np

def lu_decomposition(A: list) -> tuple:
    """
    Perform LU decomposition on a square matrix using Doolittle's method.

    Returns:
        (L, U) where L is lower triangular with 1s on diagonal,
        U is upper triangular, and A = L @ U.
    """

    A = np.array(A, dtype=float)
    n = A.shape[0]

    if A.shape != (n, n):
        raise ValueError("Matrix must be square")

    L = np.eye(n)
    U = np.zeros((n, n))

    tol = np.finfo(float).eps * n * np.linalg.norm(A, ord=np.inf)

    for i in range(n):

        # Calculate the i-th row of U
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(
                L[i, k] * U[k, j] for k in range(i)
            )

        # Check for zero or numerically unusable pivot
        if abs(U[i, i]) <= tol:
            raise ValueError("LU decomposition without pivoting failed")

        # Calculate the i-th column of L
        for j in range(i + 1, n):
            L[j, i] = (
                A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))
            ) / U[i, i]

    return L, U