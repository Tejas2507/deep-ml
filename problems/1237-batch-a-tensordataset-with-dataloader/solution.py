import torch
from torch.utils.data import TensorDataset, DataLoader

def batch_stats(X, y):
    """Wrap X and y in TensorDataset + DataLoader(batch_size=4, shuffle=False).

    Return (num_batches, first_batch_X_shape_tuple).
    """
    dataset = TensorDataset(X,y)
    loader = DataLoader(dataset, batch_size=4, shuffle=False )
    return (len(loader),tuple(next(iter(loader))[0].shape))
