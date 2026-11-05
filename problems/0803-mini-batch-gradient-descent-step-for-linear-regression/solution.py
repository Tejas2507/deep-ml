import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                       bias: float, batch_indices: list, lr: float) -> np.ndarray:

    X_batch = X[batch_indices]
    y_batch = y[batch_indices]

    B = len(batch_indices)

    # Predictions
    predictions = X_batch @ weights + bias

    # Errors
    errors = predictions - y_batch

    # Gradients
    dw = (2 / B) * (X_batch.T @ errors)
    db = (2 / B) * np.sum(errors)

    # Update weights and bias
    new_weights = weights - lr * dw
    new_bias = bias - lr * db

    return np.append(new_weights, new_bias)