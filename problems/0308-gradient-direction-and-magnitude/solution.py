import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	dic = {'magnitude':0.0,'direction':[0.0,0.0],'descent_direction':[0.0,0.0]}
	gradient = np.asarray(gradient)
	dic['magnitude'] = np.linalg.norm(np.asarray(gradient))
	if dic['magnitude'] == 0:
		return dic 
	dic['direction'] = (1/dic['magnitude'])*gradient
	dic['descent_direction'] = (-1/dic['magnitude'])*gradient
	return dic
