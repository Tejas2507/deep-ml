import numpy as np

def eye(n: int) -> np.ndarray:
    """n x n identity matrix without np.eye."""
    eye = np.asarray([[0.0 for _ in range(n)] for _ in range(n)])
    for i in range(n):
        eye[i,i] = 1.0
    return eye
    