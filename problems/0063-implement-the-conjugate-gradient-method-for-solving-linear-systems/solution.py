import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    """
    Solve Ax = b using the Conjugate Gradient method.

    :param A: Symmetric positive-definite matrix
    :param b: Right-hand side vector
    :param n: Maximum number of iterations
    :param x0: Initial guess (default is zero vector)
    :param tol: Convergence tolerance
    :return: Solution vector x
    """

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # Initial guess
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = np.array(x0, dtype=float).copy()

    # Initial residual and search direction
    r = b - A @ x
    p = r.copy()

    rs_old = np.dot(r, r)

    for _ in range(n):

        # Check convergence
        if np.sqrt(rs_old) < tol:
            break

        Ap = A @ p

        denominator = np.dot(p, Ap)

        if denominator <= 0:
            raise ValueError("A must be symmetric positive-definite")

        # Step size
        alpha = rs_old / denominator

        # Update solution
        x = x + alpha * p

        # Update residual
        r = r - alpha * Ap

        rs_new = np.dot(r, r)

        # Check convergence
        if np.sqrt(rs_new) < tol:
            break

        # Update search direction
        beta = rs_new / rs_old
        p = r + beta * p

        rs_old = rs_new

    return x