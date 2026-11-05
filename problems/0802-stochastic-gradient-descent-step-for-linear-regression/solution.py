import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
   
    for i in range(n_iter):
            j = i % len(X)
            weights -= learning_rate*2*(X[j].T@weights - y[j])*X[j]
    return weights
    
