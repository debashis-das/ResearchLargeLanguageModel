import torch
import torch.distributed as dist
from torch import nn
import triton

from config import Config
from transformer.FusedAttention import _attention
from kernels.AttentionBackwardKernel import _attention_bwd_pre_process
from transformer.RMSNorm import RMSNorm
from transformer.RopeEmbedding import RopeEmbedding
from partitioner.gpu import identify_nodes_for_qkv, nodes_partion_q_fixed_kv, nodes_partion_vary_qkv

DEVICE = triton.runtime.driver.active.get_active_torch_device()

class MultiGPUExecutor:

  def __init__(self, world_size, rank, device=DEVICE):
    self.device = device
    self.rank = rank
    self.world_size = world_size
    self.tokens_per_gpu = Config.tokens//world_size
    self.embedding = nn.Embedding(Config.total_vocab, Config.hiddens, device=DEVICE)
    self.rms = RMSNorm(Config.hiddens, device=DEVICE)
    self.W_q = nn.LazyLinear(Config.hiddens, bias=False, device=DEVICE)
    self.W_k = nn.LazyLinear(Config.hiddens, bias=False, device=DEVICE)
    self.W_v = nn.LazyLinear(Config.hiddens, bias=False, device=DEVICE)
    self.rope_embedding = RopeEmbedding(Config.hiddens, Config.dropout, self.tokens_per_gpu, rank, device=DEVICE)
    self.attention = _attention.apply

  def execute(self):
    try:
      if world_size != 1:
        exe_order_per_rank_v, exe_order_per_rank_h, exe_order_per_rank_unaligned  = identify_nodes_for_qkv(world_size)
      sample_input = torch.randint(low=0, high=Config.total_vocab, size=(self.tokens_per_gpu,)).tolist()
      input = torch.tensor(sample_input, dtype=torch.int32, device=DEVICE)
      embedded_tensor = self.embedding(input)
      X = self.rms(embedded_tensor)
      input_q, input_k, input_v = self.W_q(X), self.W_k(X), self.W_v(X)
      input_q, input_k = self.rope_embedding(input_q, input_k)
      input_q = input_q.reshape(self.tokens_per_gpu, Config.num_heads, -1).permute(1, 0, 2).contiguous()
      input_k = input_k.reshape(self.tokens_per_gpu, Config.num_heads, -1).permute(1, 0, 2).contiguous()
      input_v = input_v.reshape(self.tokens_per_gpu, Config.num_heads, -1).permute(1, 0, 2).contiguous()

      print(f"input_q{input_q.shape} strides : {input_q.stride()}")
      print(f"input_k{input_k.shape} strides : {input_k.stride()}")
      print(f"input_v{input_v.shape} strides : {input_v.stride()}")

      output_o = self.attention(input_q, input_k, input_v, Config.block_m, Config.block_n, Config.num_heads, self.tokens_per_gpu,
                                        Config.hiddens, Config.sm_scale, DEVICE, rank, rank)
      # print(f"Output ({rank},{rank}) : {output_o}")
      do = torch.rand_like(output_o)
      BLOCK_M = 32
      BLOCK_N = 16
      pre_block = 64
      num_hiddens = input_q.shape[-1]
      n_ctx = input_q.shape[1]
      grid = (input_q.shape[1]//pre_block, Config.num_heads, 1)
      print(f"Grid : {grid}, q: {input_q.shape}, k: {input_k.shape}, v: {input_v.shape} ")
      delta = torch.empty((input_q.shape[0], input_q.shape[1]), device=input_q.device, dtype=torch.float32)
      # Preprocess
      _attention_bwd_pre_process[grid](output_o, do, delta, n_ctx, pre_block, Config.num_heads, num_hiddens)
      print(f"Delta : {delta}")
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

if __name__ == "__main__":
  # device = 'cuda' if torch.cuda.is_available() else 'cpu'
  # per gpu code
  dist.init_process_group("nccl")
  world_size = dist.get_world_size()
  rank = dist.get_rank()
  multi_gpu_executor = MultiGPUExecutor(world_size, rank)
  multi_gpu_executor.execute()