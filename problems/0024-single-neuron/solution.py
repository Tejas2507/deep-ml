import math
import numpy as np 
def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	logits = [np.dot(inputs, weights) + bias for inputs in features]
	probabilities = [round(1/(1+math.exp(-z)),4) for z in logits]
	mse = sum([round((probabilities[i]-labels[i])**2,4) for i in range(len(features))])/len(features)

	
	return probabilities, mse