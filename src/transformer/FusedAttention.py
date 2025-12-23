import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _attention_forward_inner_mask(acc, l_i, m_i, q, desc_k, desc_v, 
                             offset_y, dtype: tl.constexpr, start_m, qk_scale,
                             block_m: tl.constexpr, hidden_dim: tl.constexpr, block_n: tl.constexpr, stage: tl.constexpr,
                             offs_m: tl.constexpr, offs_n: tl.constexpr, n_ctx: tl.constexpr, non_mask: tl.constexpr, warp_specialize: tl.constexpr):
    # print(f"Stage : {stage}")
    if stage == 1:
        lo, hi = 0, start_m*block_m
    elif stage == 2:
        lo, hi = start_m*block_m, (start_m+1)*block_m
    else:
        lo, hi = 0, n_ctx

    if non_mask:
        lo, hi = 0, (start_m+1)*block_m

    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo
    for start_n in tl.range(lo, hi, block_n, warp_specialize=warp_specialize):
        # print(f"start_n : {start_n} : offset of k [{offsetk_y},0]")
        k = desc_k.load([offsetk_y,0]).T
        qk = tl.dot(q, k)
        if stage == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            # print(f"mask is used {mask}")
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            # print("no mask used")
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)

        acc = acc * alpha[:, None]

        # print(f"Offset of v [0, {offsetv_y}]")
        v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += block_n
        offsetv_y += block_n
    return acc, l_i, m_i

@triton.jit
def _attention_forward(sm_scale, max_tensor, num_heads, n_ctx, desc_q, desc_k, desc_v, desc_o,
                       hidden_dim: tl.constexpr, block_m: tl.constexpr, block_n: tl.constexpr, mask_region: tl.constexpr,
                       warp_specialize: tl.constexpr):
    dtype = tl.float32
    assert block_n <= hidden_dim
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    # print(f"start_m : {start_m}, off_h : {off_h}")
    y_dim = num_heads * n_ctx
    desc_q = tl.make_tensor_descriptor(desc_q, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                     block_shape=[block_m, hidden_dim])
    desc_v = tl.make_tensor_descriptor(desc_v, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                         block_shape=[block_n, hidden_dim])
    desc_k = tl.make_tensor_descriptor(desc_k, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                     block_shape=[block_n, hidden_dim])
    desc_o = tl.make_tensor_descriptor(desc_o, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                     block_shape=[block_m, hidden_dim])
    offset_y = off_h*n_ctx
    # print(f"offset_y : {offset_y}")
    qo_offset_y = offset_y + start_m*block_m
    # print(f"qo_offset_y : {qo_offset_y}")
    offs_m = start_m*block_m + tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    # print(f"offs_m : {offs_m}")
    # print(f"offs_n : {offs_n}")
    # initialize pointer to m and l
    m_i = tl.zeros([block_m], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([block_m], dtype=tl.float32) + 1.0
    acc = tl.zeros([block_m, hidden_dim], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504 #1/log(2)
    q = desc_q.load([qo_offset_y,0])
    # print(f"q load : {[qo_offset_y, 0]}")
    if mask_region:
        acc, l_i, m_i = _attention_forward_inner_mask(acc, l_i, m_i, q, desc_k, desc_v, offset_y, dtype, start_m, qk_scale, block_m, hidden_dim, block_n, 1, offs_m, offs_n, n_ctx, False, warp_specialize)
        acc, l_i, m_i = _attention_forward_inner_mask(acc, l_i, m_i, q, desc_k, desc_v, offset_y, dtype, start_m, qk_scale, block_m, hidden_dim, block_n, 2, offs_m, offs_n, n_ctx, False, warp_specialize)
    else:
        acc, l_i, m_i = _attention_forward_inner_mask(acc, l_i, m_i, q, desc_k, desc_v, offset_y, dtype, start_m, qk_scale, block_m, hidden_dim, block_n, 2, offs_m, offs_n, n_ctx, True, warp_specialize)
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = max_tensor + off_h * n_ctx + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))     

class _attention(torch.autograd.Function):
  
  # Assumption that it is used only for causal case
  @staticmethod
  def forward(ctx, q, k, v, block_m, block_n, num_heads, n_ctx, hidden_dim, sm_scale, device, q_index, kv_index, warp_specialize=True):
      # HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
      
      # q_with_head = q.unsqueeze(0).expand(num_heads, -1, -1)
      # k_with_head = k.unsqueeze(0).expand(num_heads, -1, -1)
      # v_with_head = v.unsqueeze(0).expand(num_heads, -1, -1)
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
      
      return o
  
  @staticmethod
  def backward(ctx, do):
      return None