
from torch import nn
import torch

from lora.loRA import loRA

class loRALinear(nn.Module):
    def __init__(self, frozen_linear_layer: nn.Linear, rank: int, alpha: float, dtype: torch.dtype, device: str):
        super().__init__()
        frozen_linear_layer.requires_grad_(False)
        self.frozen_linear_layer = frozen_linear_layer
        self.loRA_module = loRA(frozen_linear_layer.in_features, frozen_linear_layer.out_features, rank, alpha, dtype, device)
    
    def load_lora_parameters(self, lora_up_proj, lora_down_proj):
        self.loRA_module.load_weights(lora_up_proj, lora_down_proj)

    def save_lora_parameters(self):
        return self.loRA_module.save_weights()

    def forward(self, X):
        frozen_output = self.frozen_linear_layer(X)
        loRA_projection = self.loRA_module(X)
        return frozen_output + loRA_projection