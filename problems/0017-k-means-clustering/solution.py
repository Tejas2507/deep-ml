
import numpy as np
def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:

	

	def dist(x,y):
		return sum([(x[i] - y[i])**2 for i in range(len(x))])
	centroids = initial_centroids
	for _ in range(max_iterations):
		clusters = [[] for _ in range(k)]
		for i in points:
			d = [dist(i,j) for j in centroids]
			clusters[d.index(min(d))].append(i)
		
		for l in range(k):
			centroid = np.mean(clusters[l],axis=0)
			centroids[l] = centroid 
		
		centroids = [tuple(centroid) for centroid in centroids]
	
	return centroids
			

			
				





	return final_centroids