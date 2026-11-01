import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Perform QR decomposition using Gram-Schmidt process.

    Returns:
        Q: Matrix with orthonormal columns
        R: Upper triangular matrix
    """

    A = np.array(A, dtype=float)

    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix")

    m, n = A.shape

    if m < n:
        raise ValueError("Number of rows must be >= number of columns")

    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    tol = np.finfo(float).eps * max(m, n) * np.linalg.norm(A, ord='fro')

    for j in range(n):

        # Start with the original j-th column
        v = A[:, j].copy()

        # Remove projections onto previously calculated Q columns
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v = v - R[i, j] * Q[:, i]

        # Normalize the remaining vector
        R[j, j] = np.linalg.norm(v)

        if R[j, j] <= tol:
            raise ValueError("Matrix columns are linearly dependent")

        Q[:, j] = v / R[j, j]

    return Q.tolist(), R.tolist()