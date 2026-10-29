import numpy as np







def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	x = np.asarray(features)
	w = np.asarray(initial_weights)
	b = np.asarray(initial_bias)
	y = np.asarray(labels)
	n = len(x)
	mse_values = []
	for _ in range(epochs):
		z = np.dot(x, w) + b 
		pred = 1/(1+np.exp(-z))
		mse = np.mean(np.square(pred - y))
		mse_values.append(round(mse,4))

		delta = (2/n)*(pred-y)*pred*(1-pred)
		dw = np.dot(delta,x)
		db = delta.sum()

		w = w - learning_rate*dw 
		b = b - learning_rate*db 


	
	return w, b, mse_values