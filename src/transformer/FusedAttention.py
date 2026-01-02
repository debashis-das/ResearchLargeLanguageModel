import torch
import gc

from kernels.AttentionForwardKernel import _attention_forward
from kernels.AttentionBackwardKernel import _attention_bwd_pre_process, _attention_bwd

class _attention(torch.autograd.Function):
  
  # Assumption that it is used only for causal case
  @staticmethod
  def forward(ctx, q, k, v, block_m, block_n, num_heads, n_ctx, hidden_dim, sm_scale, device, q_index, kv_index, warp_specialize=True):
      # HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
      # q = q.unsqueeze(0).expand(num_heads, -1, -1)
      # k = k.unsqueeze(0).expand(num_heads, -1, -1)
      # v = v.unsqueeze(0).expand(num_heads, -1, -1)
      o = torch.empty_like(q)
      # print(q_with_head.shape, k_with_head.shape, v_with_head.shape)
      M = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
      # print(f"Max tokens : {M.shape}")
      grid = (n_ctx//block_m, num_heads, 1)
      # print(f"Grid : {grid}")
      # Attention forward
      if q_index == kv_index:
          # mask region
          _attention_forward[grid](sm_scale, M, num_heads, n_ctx,
                          q, k, v, o,
                          hidden_dim, block_m, block_n, True, warp_specialize)
      else:
          # non-mask region
          _attention_forward[grid](sm_scale, M, num_heads, q.shape[1],
                          q, k, v, o,
                          hidden_dim, block_m, block_n, False, warp_specialize)
      ctx.save_for_backward(q,k,v,o,M)
      ctx.sm_scale = sm_scale
      ctx.hidden_dim = hidden_dim
      ctx.q_index = q_index
      ctx.kv_index = kv_index
      ctx.num_heads = num_heads
      return o
  
  @staticmethod
  def backward(ctx, do):
      q, k, v, o, M = ctx.saved_tensors
      sm_scale = ctx.sm_scale
      q_index = ctx.q_index
      kv_index = ctx.kv_index
      num_heads = ctx.num_heads
      block_m = 32
      block_n = 16
      pre_block = 64
      num_hiddens = q.shape[-1]
      n_ctx = q.shape[1]
      grid_preprocess = (n_ctx//pre_block, num_heads, 1)
      print(f"Grid (bwd_pre_process) : {grid_preprocess}, q: {q.shape}, k: {k.shape}, v: {v.shape} ")
      delta = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
      # Preprocess
      _attention_bwd_pre_process[grid_preprocess](o, do, delta, n_ctx, pre_block, num_heads, num_hiddens)
      # bwd
      dq = torch.empty_like(q)
      dk = torch.empty_like(k)
      dv = torch.empty_like(v)
      bulk_slice_factor = 1
      grid_bwd = (n_ctx//block_m, num_heads, 1)
      print(f"Grid (bwd) : {grid_bwd}")
      gc.collect()
      torch.cuda.empty_cache()
      _attention_bwd[grid_bwd](q, k, v, do, dq, dk, dv, M, delta, sm_scale, num_heads, n_ctx, 
                               num_hiddens, block_m, block_n, bulk_slice_factor)
      print(f"dv : {dv}")
      print(f"dk : {dk}")
      print(f"dq : {dq}")