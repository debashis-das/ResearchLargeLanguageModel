import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, num_hiddens, device, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_hiddens, device=device))
        self.variance_epsilon = eps
    
    # @torch.compile
    def forward(self, hidden_state: torch.Tensor):
        input_dtype = hidden_state.dtype
        hidden_state = hidden_state.to(torch.float32)
        variance = hidden_state.pow(2).mean(-1, keepdim=True)
        hidden_state = hidden_state*torch.rsqrt(variance+self.variance_epsilon)
        return self.weight*hidden_state.to(input_dtype)