from torch import nn

from app.config import Config


class TransformerLayer(nn.Module):

      def __init__(self, world_size, rank, tokens_per_gpu, rope_embedding=None, attention=None, W_q=None, W_k=None, W_v=None, W_down=None, mlp=None, rms1=None, rms2=None):
          super().__init__()
          self.world_size = world_size
          self.rank = rank
          self.tokens_per_gpu = tokens_per_gpu
          self.rope_embedding = rope_embedding
          self.attention = attention
          self.W_q = W_q
          self.W_k = W_k
          self.W_v = W_v
          self.W_down = W_down
          self.mlp = mlp
          self.rms1 = rms1
          self.rms2 = rms2

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
