
from torch import nn
import torch

from lora.loRA import loRA

class loRALinear(nn.Module):
    def __init__(self, frozen_linear_layer: nn.Linear, rank: int, alpha: float, device: str):
        super().__init__()
        frozen_linear_layer.requires_grad_(False)
        self.frozen_linear_layer = frozen_linear_layer
        # Trainable LoRA weights are kept in fp32 regardless of the frozen model's dtype:
        # AdamW's exp_avg_sq buffer is created in the parameter's own dtype, and squaring
        # a fp16 gradient overflows fp16's ~65504 max, silently turning the optimizer state
        # (and then the weights) into inf/nan on the very first step.
        self.loRA_module = loRA(frozen_linear_layer.in_features, frozen_linear_layer.out_features, rank, alpha, torch.float32, device)

    def forward(self, X):
        frozen_output = self.frozen_linear_layer(X)
        loRA_projection = self.loRA_module(X.float())
        return frozen_output + loRA_projection.to(frozen_output.dtype)