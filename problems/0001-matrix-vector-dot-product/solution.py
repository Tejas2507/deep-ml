import numpy as np 
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	a = np.asarray(a)
	b = np.asarray(b)
	
	if a.shape[-1] == b.shape[0]:
		return a @ b
	else:
		return -1
	pass