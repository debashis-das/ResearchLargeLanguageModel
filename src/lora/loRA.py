import math
from typing import Any
from torch import nn
import torch

class loRA(nn.Module):
    """LoRA module for low-rank adaptation of large language models. This module consists of two linear layers: an up projection and a down projection.
    The up projection maps the input to a lower-dimensional space defined by the rank, and the down projection maps it back to the original output space.
    The output of the down projection is scaled by alpha/rank to control the contribution of the LoRA module to the overall model output."""
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 1.0, dtype: torch.dtype = torch.float32, device=None):
        super().__init__()
        self.alpha = alpha
        self.rank = rank
        self.up_proj = nn.Linear(in_features, rank, bias=False, dtype=dtype, device=device)
        self.down_proj = nn.Linear(rank, out_features, bias=False, dtype=dtype, device=device)
        nn.init.zeros_(self.down_proj.weight)
        nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))

    def forward(self, X):
        projection = self.down_proj(self.up_proj(X))
        return (self.alpha / self.rank) * projection