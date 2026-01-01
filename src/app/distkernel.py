import torch
import torch.distributed as dist
from torch import nn
import triton
import gc

from config import Config
from transformer.FusedAttention import _attention
from kernels.AttentionForwardKernel import _attention_forward
from kernels.AttentionBackwardKernel import _attention_bwd_pre_process, _attention_bwd
from transformer.RMSNorm import RMSNorm
from transformer.RopeEmbedding import RopeEmbedding
from partitioner.gpu import identify_nodes_for_qkv, nodes_partion_q_fixed_kv, nodes_partion_vary_qkv

torch.set_printoptions(profile="full")
DEVICE = triton.runtime.driver.active.get_active_torch_device()

def alloc_fn(size: int, align: int, _):
  return torch.empty(size, dtype=torch.int8aaq
                     , device=DEVICE)

triton.set_allocator(alloc_fn)

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
      q, k, v = self.W_q(X), self.W_k(X), self.W_v(X)
      q, k = self.rope_embedding(q, k)
      q = q.reshape(self.tokens_per_gpu, Config.num_heads, -1).permute(1, 0, 2).contiguous()
      k = k.reshape(self.tokens_per_gpu, Config.num_heads, -1).permute(1, 0, 2).contiguous()
      v = v.reshape(self.tokens_per_gpu, Config.num_heads, -1).permute(1, 0, 2).contiguous()

      n_ctx = self.tokens_per_gpu
      num_hiddens = Config.hiddens
      o = torch.empty_like(q)
      M = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
      block_m = 32
      block_n = 16
      pre_block = 64
      grid_fwd = (n_ctx//block_m, Config.num_heads, 1)
      print(f"Grid (fwd) : {grid_fwd} : q{q.shape} strides : {q.stride()} : k{k.shape} strides : {k.stride()} : v{v.shape} strides : {v.stride()}")
      _attention_forward[grid_fwd](Config.sm_scale, M, Config.num_heads, n_ctx,
                      q, k, v, o,
                      num_hiddens, block_m, block_n, True, True)
      print(f"Output ({rank},{rank}) : {o}")
      print(f"Max tensor ({rank},{rank}) : {M}")
      do = torch.rand_like(o)
      n_ctx = q.shape[1]
      grid_preprocess = (q.shape[1]//pre_block, Config.num_heads, 1)
      print(f"Grid (Preprocess) : {grid_preprocess}, q: {q.shape}, k: {k.shape}, v: {v.shape} ")
      delta = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
      # Preprocess
      _attention_bwd_pre_process[grid_preprocess](o, do, delta, n_ctx, pre_block, Config.num_heads, num_hiddens)
      print(f"Delta : {delta}")
      dq = torch.empty_like(q)
      dk = torch.empty_like(k)
      dv = torch.empty_like(v)
      bulk_slice_factor = 1
      grid_bwd = (n_ctx//block_m, Config.num_heads, 1)
      print(f"Grid (bwd) : {grid_bwd}")
      gc.collect()
      torch.cuda.empty_cache()
      _attention_bwd[grid_bwd](q, k, v, do, dq, dk, dv, M, delta, Config.sm_scale, Config.num_heads, n_ctx, 
                               num_hiddens, block_m, block_n, bulk_slice_factor)
      print(f"dv : {dv}")
      print(f"dk : {dk}")
      print(f"dq : {dq}")
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