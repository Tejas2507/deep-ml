import numpy as np
from scipy.optimize import linprog


def lad_fit(X, y):
    """
    Fit Least Absolute Deviations regression using Linear Programming.
    Returns the optimal weight vector.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    m, d = X.shape

    # Objective: minimize sum of slack variables
    c = np.concatenate([np.zeros(d), np.ones(m)])

    # Constraints:
    # Xw - e <= y
    # -Xw - e <= -y

    A_ub = np.vstack([
        np.hstack([X, -np.eye(m)]),
        np.hstack([-X, -np.eye(m)])
    ])

    b_ub = np.concatenate([y, -y])

    # Weights are unrestricted; slack variables are nonnegative
    bounds = [(None, None)] * d + [(0, None)] * m

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method='highs'
    )

    if not result.success:
        raise ValueError(result.message)

    return result.x[:d]


def huber_fit(X, y, delta, n_iter):
    """
    Fit Huber regression using Iteratively Reweighted Least Squares.
    Returns the estimated weight vector.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    m, d = X.shape

    if delta <= 0:
        raise ValueError("delta must be positive")

    # Initialize with ordinary least squares
    weights = np.linalg.lstsq(X, y, rcond=None)[0]

    for _ in range(n_iter):

        # Calculate residuals
        residuals = X @ weights - y

        # Calculate IRLS weights
        sample_weights = np.ones(m)

        mask = np.abs(residuals) > delta

        sample_weights[mask] = (
            delta / np.abs(residuals[mask])
        )

        # Weighted least squares
        sqrt_weights = np.sqrt(sample_weights)

        X_weighted = X * sqrt_weights[:, None]
        y_weighted = y * sqrt_weights

        new_weights = np.linalg.lstsq(
            X_weighted,
            y_weighted,
            rcond=None
        )[0]

        # Convergence check
        if np.linalg.norm(new_weights - weights) < 1e-10:
            weights = new_weights
            break

        weights = new_weights

    return weights