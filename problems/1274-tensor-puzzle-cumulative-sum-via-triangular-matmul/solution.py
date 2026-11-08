import numpy as np

def cumsum(a: np.ndarray) -> np.ndarray:
    """Inclusive cumulative sum of 1-D array a without np.cumsum."""
    i = np.arange(len(a))
    utm = (i[:,None]<=i[None,:]).astype(float)
    return a@utm
