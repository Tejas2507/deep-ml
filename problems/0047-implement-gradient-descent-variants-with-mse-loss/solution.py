import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs,
                     batch_size=1, method='batch'):

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).copy()

    m, n = X.shape

    if method not in ['batch', 'stochastic', 'mini_batch']:
        raise ValueError("Invalid method")

    for _ in range(n_epochs):

        if method == 'batch':
            batches = [np.arange(m)]

        elif method == 'stochastic':
            batches = [np.array([i]) for i in range(m)]

        elif method == 'mini_batch':
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")

            batches = [
                np.arange(i, min(i + batch_size, m))
                for i in range(0, m, batch_size)
            ]

        for indices in batches:
            X_batch = X[indices]
            y_batch = y[indices]

            errors = X_batch @ weights - y_batch

            gradient = (2 / len(indices)) * (X_batch.T @ errors)

            weights -= learning_rate * gradient

    return weights