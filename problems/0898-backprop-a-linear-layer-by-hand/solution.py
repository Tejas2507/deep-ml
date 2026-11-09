import torch

def linear_backward(grad_output, x, W):
    
    grad_input = torch.matmul(grad_output,W)
    grad_W = torch.matmul(grad_output.T,x)
    grad_b = torch.sum(grad_output,axis=0)
    return grad_input,grad_W,grad_b
