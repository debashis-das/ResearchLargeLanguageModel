import torch
import torch.distributed as dist
from torch import nn
import triton
import gc

from config import Config
from transformer.MLP import MLP
from transformer.FusedAttention import _attention
from transformer.RMSNorm import RMSNorm
from transformer.RopeEmbedding import RopeEmbedding
from partitioner.gpu import identify_nodes_for_qkv, nodes_partion_q_fixed_kv, nodes_partion_vary_qkv

# torch.set_printoptions(profile="full")
DEVICE = triton.runtime.driver.active.get_active_torch_device()


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
    self.loss_fn = nn.CrossEntropyLoss()

  def execute(self, tokens):
    try:
      if world_size != 1:
        exe_order_per_rank_v, exe_order_per_rank_h, exe_order_per_rank_unaligned  = identify_nodes_for_qkv(world_size)
      shift_labels, shit_logits = self.transformerPerGPU(tokens)
      

      # if world_size != 1:
      #   exe_order_per_rank_v[rank].remove((rank,rank))
      #   exe_order_per_rank_h[rank].remove((rank,rank))

      #   dist.barrier()
      #   nodes_partion_q_fixed_kv(exe_order_per_rank_h, exe_order_per_rank_v,
      #                           self.rank, input_q, input_k, input_v)
      #   dist.barrier()
      #   nodes_partion_vary_qkv(exe_order_per_rank_unaligned, self.rank,
      #                         input_q, input_k, input_v)
      #   dist.barrier()
    finally:
      dist.destroy_process_group()

  def transformerPerGPU(self, tokens):
      input = torch.tensor(tokens, dtype=torch.int32, device=DEVICE)
      print(f"Input : {input.shape}")
      X = self.embedding(input)
      for _ in range(Config.learning_iter):
        X = self.rms1(X)
        q, k, v = self.W_q(X), self.W_k(X), self.W_v(X)
        q, k = self.rope_embedding(q, k)
        q = q.reshape(self.tokens_per_gpu, Config.num_heads, -1).permute(1, 0, 2).contiguous()
        k = k.reshape(self.tokens_per_gpu, Config.num_heads, -1).permute(1, 0, 2).contiguous()
        v = v.reshape(self.tokens_per_gpu, Config.num_heads, -1).permute(1, 0, 2).contiguous()

        n_ctx = self.tokens_per_gpu
        block_m = 32
        block_n = 16
        grid_fwd = (n_ctx//block_m, Config.num_heads, 1)
        print(f"Grid (fwd) : {grid_fwd} : q{q.shape} strides : {q.stride()} : k{k.shape} strides : {k.stride()} : v{v.shape} strides : {v.stride()}")
        output = self.attention(q, k, v, block_m, block_n, Config.num_heads, n_ctx, Config.hiddens, Config.sm_scale, DEVICE, self.rank, self.rank)
        print(f"Output ({rank},{rank}) : {output.shape}")
        output = output.permute(1,0,2).reshape(self.tokens_per_gpu,-1)
        v = v.permute(1,0,2).reshape(self.tokens_per_gpu, -1)
        print(f"Output after permute & reshape ({rank},{rank}) o:{output.shape}, v:{v.shape}")
        x_residual = output + v
        y_rms = self.rms2(x_residual)
        z = self.mlp(y_rms)
        X = x_residual + self.W_down(z)
      X = self.rms3(X)
      logits = self.dense(X)
      print(f"Logits : {logits.shape}")
      logits = logits.float()
      print(f"Logits : {logits}")
      shift_labels = input[1:].contiguous()
      shift_logits = logits[:-1,:].contiguous()
      return shift_labels, shift_logits



if __name__ == "__main__":
  # device = 'cuda' if torch.cuda.is_available() else 'cpu'
  # per gpu code
  dist.init_process_group("nccl")
  world_size = dist.get_world_size()
  tokens_per_gpu = Config.tokens//world_size

  rank = dist.get_rank()
  multi_gpu_executor = MultiGPUExecutor(world_size, rank)
  tokens = torch.randint(low=0, high=Config.total_vocab, size=(tokens_per_gpu,)).tolist()
  logits = multi_gpu_executor.execute(tokens)
