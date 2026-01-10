import torch
import torch.distributed as dist
from torch import nn
import triton

from config import Config
from transformer.MLP import MLP
from transformer.FusedAttentionBatch import _attention
from transformer.RMSNorm import RMSNorm
from transformer.RopeEmbedding import RopeEmbedding

# torch.set_printoptions(profile="full")
DEVICE = triton.runtime.driver.active.get_active_torch_device()
# DEVICE = "cpu"


class MultiGPUExecutor:

  def __init__(self, world_size, rank, device=DEVICE):
    self.device = device
    self.rank = rank
    self.world_size = world_size
    self.tokens_per_gpu = Config.tokens//world_size
    self.embedding = nn.Embedding(Config.total_vocab, Config.hiddens, device=DEVICE)
    self.rms1 = RMSNorm(Config.hiddens, device=DEVICE)
    self.rms2 = RMSNorm(Config.hiddens, device=DEVICE)
    self.rms3 = RMSNorm(Config.hiddens, device=DEVICE)
    self.mlp = MLP(Config.hiddens, Config.mlp_intermediate_hidden, device=DEVICE)

    self.W_q = nn.LazyLinear(Config.hiddens, bias=False, device=DEVICE)
    self.W_k = nn.LazyLinear(Config.hiddens, bias=False, device=DEVICE)
    self.W_v = nn.LazyLinear(Config.hiddens, bias=False, device=DEVICE)
    self.W_down = nn.LazyLinear(Config.hiddens, bias=False, device=DEVICE)
    self.dense = nn.LazyLinear(Config.total_vocab, bias=False, device=DEVICE)

    self.rope_embedding = RopeEmbedding(Config.hiddens, Config.dropout, self.tokens_per_gpu, rank, device=DEVICE)
    self.attention = _attention.apply
    self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

  def execute(self, tokens):
    try:
      loss = self.transformerPerGPU(tokens)
      dist.all_reduce(loss, op=dist.ReduceOp.SUM)
      loss = loss / (Config.batch*Config.tokens)
      print(f"Loss : {loss}")
      loss.backward()
    finally:
      dist.destroy_process_group()

  def transformerPerGPU(self, tokens):
      src_tokens = torch.tensor(tokens, dtype=torch.int32, device=DEVICE)
      X = self.embedding(src_tokens)
      for _ in range(1):
        # print(f"X shape : {X.shape}")
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
        output = self.attention(q, k, v, block_m, block_n, Config.batch, Config.num_heads, n_ctx, Config.hiddens, 
                                Config.sm_scale, world_size, self.rank)
        # print(f"Output ({rank},{rank}): {output.shape}")
        output = output.permute(0, 2, 1, 3).reshape(Config.batch, self.tokens_per_gpu,-1)
        v = v.permute(0, 2, 1, 3).reshape(Config.batch, self.tokens_per_gpu, -1)
        # print(f"Output after permute & reshape ({rank},{rank}) o:{output.shape}, v:{v.shape}")
        x_residual = output + v
        # print(f"x_residual : {output.shape}, {v.shape}, {x_residual.shape}")
        y_rms = self.rms2(x_residual)
        z = self.mlp(y_rms)
        X = x_residual + self.W_down(z)
      X = self.rms3(X)
      logits = self.dense(X)
      logits = logits.float()
      # print(f"Logits : {logits.shape} : {logits}")
      B, T, H = logits.shape
      logits = logits.view(B*T, H)
      src_tokens = src_tokens.view(B*T)
      shift_labels = src_tokens[...,1:].contiguous()
      shift_logits = logits[...,:-1,:].contiguous()
      # print(f"shift_logits: {shift_logits.shape}, shift_labels: {shift_labels.shape}")
      loss = self.loss_fn(shift_logits, shift_labels.long())
      return loss

if __name__ == "__main__":
  # device = 'cuda' if torch.cuda.is_available() else 'cpu'
  # per gpu code
  # dist.init_process_group("gloo")
  dist.init_process_group("nccl")

  world_size = dist.get_world_size()
  tokens_per_gpu = Config.tokens//world_size

  rank = dist.get_rank()
  multi_gpu_executor = MultiGPUExecutor(world_size, rank)
  tokens = torch.randint(low=0, high=Config.total_vocab, size=(Config.batch, tokens_per_gpu,)).tolist()
  multi_gpu_executor.execute(tokens)
