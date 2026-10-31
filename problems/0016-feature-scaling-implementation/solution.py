import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	mean = np.mean(data,axis=0)
	
	std = np.std(data,axis=0)
	
	standardized_data = np.round((data-mean)/std,4)


	mmin = np.min(data,axis=0)
	mmax = np.max(data,axis=0)
	ranger = mmax-mmin
	normalized_data = np.round((data - mmin)/ranger,4)


	return standardized_data, normalized_data