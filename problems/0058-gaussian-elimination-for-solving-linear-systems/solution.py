import numpy as np

def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.

    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """

    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)

    n = A.shape[0]

    if A.shape != (n, n):
        raise ValueError("A must be a square matrix")

    original_shape = b.shape
    b = b.reshape(-1)

    if len(b) != n:
        raise ValueError("Incompatible dimensions")

    tol = np.finfo(float).eps * n * np.linalg.norm(A, ord=np.inf)

    # Forward elimination with partial pivoting
    for i in range(n):

        # Find the row with the largest absolute pivot
        pivot = i + np.argmax(np.abs(A[i:, i]))

        # Check for singularity
        if abs(A[pivot, i]) <= tol:
            raise ValueError("Matrix is singular or nearly singular")

        # Swap rows if necessary
        if pivot != i:
            A[[i, pivot]] = A[[pivot, i]]
            b[[i, pivot]] = b[[pivot, i]]

        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]

            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Back substitution
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    return x.reshape(original_shape) if len(original_shape) == 2 else x