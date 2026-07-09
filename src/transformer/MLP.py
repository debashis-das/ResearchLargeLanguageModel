from torch import nn
# import torch

class MLP(nn.Module):
    """The MLP network."""
    def __init__(self, num_hiddens, num_intermediate, device):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_intermediate = num_intermediate
        self.gate_proj = nn.Linear(self.num_hiddens, self.num_intermediate, bias=False, device=device)
        self.up_proj = nn.Linear(self.num_hiddens, self.num_intermediate, bias=False, device=device)
        self.down_proj = nn.Linear(self.num_intermediate, self.num_hiddens, bias=False, device=device)
        self.act_fn = nn.SiLU()

    # @torch.compile(mode="max-autotune")
    def forward(self, X):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(X)) * self.up_proj(X))
        return down_proj