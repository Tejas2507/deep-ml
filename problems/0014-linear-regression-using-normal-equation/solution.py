import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	X = np.asarray(X)
	y = np.asarray(y)
	theta = np.linalg.inv(X.T @ X) @ X.T @ y  
	theta = [round(t,4) for t in theta]
	return theta