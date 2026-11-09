import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features,out_features)
        # TODO: call the parent constructor and register an nn.Linear as self.linear 

        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.linear(x)
        return output 
        
