import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	scores = np.asarray(scores)
	summ = np.sum(np.exp(scores - max(scores)))
	return scores - max(scores) - np.log(summ)

	pass