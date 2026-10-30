import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    w = initial_weights.clone().detach().requires_grad_(True)
    b = torch.tensor(initial_bias, dtype=features.dtype, device=features.device,requires_grad=True)

    optimizer = torch.optim.SGD([w,b],lr=learning_rate)
    mse_values = []

    for _ in range(epochs):
        optimizer.zero_grad()
        z = torch.matmul(features, w) + b 
        pred = torch.sigmoid(z)

        loss = nn.functional.mse_loss(pred,labels.float())
        loss.backward()
        optimizer.step()

        mse_values.append(round(loss.item(),4))

    updated_weights = [round(wi.item(), 4) for wi in w]
    updated_bias = round(b.item(), 4)

    return updated_weights, updated_bias, mse_values
    








    pass