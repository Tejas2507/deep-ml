import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # TODO: save anything backward will need, then return the ReLU output
        ctx.save_for_backward((x>0).to(x.dtype))
        return torch.clamp(x,min=0)
      

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: return dL/dx using the saved tensors and grad_output
        (mask,) = ctx.saved_tensors
        return mask * grad_output

        
