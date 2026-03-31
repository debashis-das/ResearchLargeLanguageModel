from app.distkernel import DEVICE
from torch import nn

from app.config import Config
from transformer import RopeEmbedding
from transformer.FusedAttentionBatch import _attention
from transformer.MLP import MLP
from transformer.RMSNorm import RMSNorm


class TransformerLayer(nn.Module):

      def __init__(self, world_size, rank, tokens_per_gpu, device):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        self.tokens_per_gpu = tokens_per_gpu
        self.rms1 = RMSNorm(Config.hiddens, device=device)
        self.rms2 = RMSNorm(Config.hiddens, device=device)
        self.mlp = MLP(Config.hiddens, Config.mlp_intermediate_hidden, device=device)

        self.W_q = nn.LazyLinear(Config.hiddens, bias=False, device=device)
        self.W_k = nn.LazyLinear(Config.hiddens, bias=False, device=device)
        self.W_v = nn.LazyLinear(Config.hiddens, bias=False, device=device)
        self.W_down = nn.LazyLinear(Config.hiddens, bias=False, device=device)

        self.rope_embedding = RopeEmbedding(Config.hiddens, Config.dropout, tokens_per_gpu, rank, device=device)
        self.attention = _attention.apply

      def forward(self, X):
        X = self.rms1(X)
        q, k, v = self.W_q(X), self.W_k(X), self.W_v(X)
        q, k = self.rope_embedding(q, k) 
        q = q.reshape(Config.batch, self.tokens_per_gpu, Config.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(Config.batch, self.tokens_per_gpu, Config.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(Config.batch, self.tokens_per_gpu, Config.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        n_ctx = self.tokens_per_gpu
        block_m = 32
        block_n = 16
        grid_fwd = (n_ctx//block_m, Config.num_heads*Config.batch, 1)
        # print(f"Grid (fwd) : {grid_fwd} : q{q.shape} strides : {q.stride()} : k{k.shape} strides : {k.stride()} : v{v.shape} strides : {v.stride()}")
        output = self.attention(q, k, v, Config.batch, Config.num_heads, n_ctx, Config.hiddens, 
                                Config.sm_scale, self.world_size, self.rank)
        # print(f"Output ({rank},{rank}): {output.shape} : {output[:,:,:10,:10]}")
        output = output.permute(0, 2, 1, 3).reshape(Config.batch, self.tokens_per_gpu,-1)
        v = v.permute(0, 2, 1, 3).reshape(Config.batch, self.tokens_per_gpu, -1)
        # print(f"Output after permute & reshape ({rank},{rank}) o:{output.shape}, v:{v.shape}")
        x_residual = output + v
        # print(f"x_residual : {output.shape}, {v.shape}, {x_residual.shape}")
        y_rms = self.rms2(x_residual)
        z = self.mlp(y_rms)
        # print(f"z : {z.shape}, {z[:,:10,:10]}")
        X = x_residual + self.W_down(z)
        # print(f"X after MLP and residual : {X.shape}, {X}")
        return X
