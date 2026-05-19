# import torch
import triton
import triton.language as tl

@triton.jit
def _attention_bwd_pre_process(o_ptr, do_ptr, delta_ptr,
                               batch: tl.constexpr,
                               n_ctx: tl.constexpr,
                               pre_block: tl.constexpr,
                               heads: tl.constexpr,
                               hidden: tl.constexpr):
    pre_block_per_nctx = tl.program_id(0)
    off_hz = tl.program_id(1)
    head_idx = off_hz % heads
    batch_idx = off_hz // heads
    offs_pre_block = pre_block_per_nctx*pre_block + tl.arange(0, pre_block)
    offs_hid = tl.arange(0, hidden)
    offset = batch_idx*heads*n_ctx*hidden + head_idx*n_ctx*hidden + offs_pre_block[:,None]*hidden + offs_hid[None,:]
    o = tl.load(o_ptr + offset)
    do = tl.load(do_ptr + offset)
    o = o.to(tl.float32)
    do = do.to(tl.float32)
    # o = tl.maximum(tl.minimum(o, 1.0e8), -1.0e8)
    # do = tl.maximum(tl.minimum(do, 1.0e8), -1.0e8)
    o_do = tl.sum(o*do, axis=1)
    delta = delta_ptr + batch_idx*heads*n_ctx + head_idx*n_ctx + offs_pre_block
    tl.store(delta, o_do)

@triton.jit
def _attention_bwd_dkdv(dkey, dvalue, m, d, q, k, v, do,
                        init_offset, offset_along_n, offset_along_h, block_m, block_n, batch_idx, head_idx, ctxid, num_heads, n_ctx, hidden_dim, sm_scale, causal=False, mask=False):
    base_mask = head_idx*n_ctx + batch_idx*num_heads*n_ctx
    mask_offset_along_m = ctxid*block_m + base_mask + tl.arange(0, block_m)
    mask_offset_along_n = ctxid*block_m + base_mask + tl.arange(0, block_n)
    if causal and mask:
      num_steps = block_m // block_n
    elif causal and not mask:
      num_steps = (n_ctx - (ctxid+1)*block_m) // block_n
      mask_offset_along_n = (ctxid+1)*block_m + base_mask + tl.arange(0, block_n)
    else:
      num_steps = n_ctx // block_n
    
    offset_block_n = init_offset + offset_along_n[:, None] + offset_along_h[None, :]
    offset_block_n_T = init_offset + offset_along_n[None, :] + offset_along_h[:, None]
    for _ in range(num_steps):
      queryT = tl.load(q + offset_block_n_T)  # pre-load to L1
      d_of_o = tl.load(do + offset_block_n)  # pre-load to L1
      max_tensor = tl.load(m + mask_offset_along_n)  # pre-load to L1
      kqT = tl.dot(k, queryT)*sm_scale
      pT = tl.exp(kqT - max_tensor[None,:])
      if mask:
        mask_tensor = (mask_offset_along_m[:, None] <= mask_offset_along_n[None, :])
        pT += tl.where(mask_tensor, 0, 0.0)

      dvalue += tl.dot(pT, d_of_o).to(tl.float32)
      dpT = tl.dot(v, tl.trans(d_of_o)).to(tl.float32)
      delta = tl.load(d + mask_offset_along_n)  # pre-load to L1
      dsT = pT * (dpT - delta[None,:])
      dkey += tl.dot(dsT, tl.trans(queryT)).to(tl.float32)

      mask_offset_along_n += block_n
      offset_block_n += block_n*hidden_dim
      offset_block_n_T += block_n*hidden_dim
    return dkey, dvalue

@triton.jit
def _attention_bwd_dq(dquery, m, d, q, k, v, do,
                        offset_batch_head, offset_along_n, offset_along_h, block_m, block_n, batch_idx, head_idx, ctxid, num_heads, n_ctx, hidden_dim, sm_scale, causal=False, mask=False):
    if causal and not mask and ctxid == 0:
      return dquery
    base_mask = head_idx*n_ctx + batch_idx*num_heads*n_ctx
    mask_offset_along_m = ctxid*block_m + base_mask + tl.arange(0, block_m)
    if causal and mask:
      num_steps = block_m // block_n
      mask_offset_along_n = ctxid*block_m + base_mask + tl.arange(0, block_n)
      offset_block_n_T = ctxid*block_m*hidden_dim + offset_batch_head + offset_along_n[None, :] + offset_along_h[:, None]
    elif causal and not mask:
      num_steps = ctxid*block_m // block_n
      offset_block_n_T = offset_batch_head + offset_along_n[None, :] + offset_along_h[:, None]
    else:
      num_steps = n_ctx // block_n
    
    max_tensor = tl.load(m + mask_offset_along_m)  # pre-load to L1
    delta = tl.load(d + mask_offset_along_m)  # pre-load to L1
    
    for _ in range(num_steps):
      keyT = tl.load(k + offset_block_n_T)  # pre-load to L1
      valueT = tl.load(v + offset_block_n_T)  # pre-load to L1
      qkT = tl.dot(q, keyT)*sm_scale
      p = tl.exp(qkT - max_tensor[:, None])
      if mask:
        mask_tensor = (mask_offset_along_m[:, None] >= mask_offset_along_n[None, :])
        p += tl.where(mask_tensor, 0, 0.0)
        # increment only in case of mask
        mask_offset_along_n += block_n
      dp = tl.dot(do, valueT).to(tl.float32)
      ds = p * (dp - delta[:, None])
      dquery += tl.dot(ds, tl.trans(keyT))
      offset_block_n_T += block_n*hidden_dim
    return dquery

# @triton.autotune(
#     configs=[
#         triton.Config({'block_m':32, 'block_n':32}, num_warps=4, num_stages=1),
#         triton.Config({'block_m':32, 'block_n':16}, num_warps=4, num_stages=1),
#         triton.Config({'block_m':64, 'block_n':32}, num_warps=4, num_stages=1),
#         triton.Config({'block_m':64, 'block_n':16}, num_warps=4, num_stages=1)
#     ],
#     key=['n_ctx', 'hidden_dim'],   # runtime-dependent shapes
# )

  
@triton.jit
def _attention_bwd(q, k, v, do, dq, dk, dv, m, d,
                   sm_scale: tl.constexpr, batch: tl.constexpr, num_heads: tl.constexpr,
                   n_ctx: tl.constexpr, hidden_dim: tl.constexpr, block_m: tl.constexpr,
                   block_n: tl.constexpr, CAUSAL: tl.constexpr = True):
    # LN2 = 0.6931471824645996  # = ln(2)
    # current context block
    ctxid = tl.program_id(0)
    # current head
    hzid = tl.program_id(1)
    batch_idx = hzid // num_heads
    head_idx = hzid % num_heads
    # init offset can be used for both dkdv & dq
    offset_batch_head = head_idx*n_ctx*hidden_dim + batch_idx*num_heads*n_ctx*hidden_dim
    init_offset = ctxid*block_m*hidden_dim + offset_batch_head

    dvalue = tl.zeros([block_m, hidden_dim], dtype=tl.float32)
    dkey = tl.zeros([block_m, hidden_dim], dtype=tl.float32)

    offset_along_m = tl.arange(0, block_m)*hidden_dim
    offset_along_n = tl.arange(0, block_n)*hidden_dim
    offset_along_h = tl.arange(0, hidden_dim)
    
    offset_block_m = init_offset + offset_along_m[:, None] + offset_along_h[None, :]
    key = tl.load(k + offset_block_m)
    value = tl.load(v + offset_block_m)
    assert block_m % block_n == 0, "block_m should be divisible by block_n"

    if CAUSAL:
      # for the mask part of block_m
      dkey, dvalue = _attention_bwd_dkdv(dkey, dvalue, m, d, q, key, value, do, init_offset,
                          offset_along_n, offset_along_h, block_m, block_n, batch_idx, head_idx, ctxid, num_heads, n_ctx, hidden_dim, sm_scale, causal=True, mask=True)
      # for the non-mask part for data before the mask (past data)
      dkey, dvalue = _attention_bwd_dkdv(dkey, dvalue, m, d, q, key, value, do, init_offset,
                          offset_along_n, offset_along_h, block_m, block_n, batch_idx, head_idx, ctxid, num_heads, n_ctx, hidden_dim, sm_scale, causal=True, mask=False)
    else:
      # for non-causal, we feed both the past and future data together as there is no mask
      dkey, dvalue = _attention_bwd_dkdv(dkey, dvalue, m, d, q, key, value, do, init_offset,
                          offset_along_n, offset_along_h, block_m,block_n, batch_idx, head_idx, ctxid, num_heads, n_ctx, hidden_dim, sm_scale, causal=False, mask=False)
    
    tl.store(dv + offset_block_m, dvalue)
    tl.store(dk + offset_block_m, dkey*sm_scale)  # scale back the dkey as we had scaled the kqT in forward

    dquery = tl.zeros([block_m, hidden_dim], dtype=tl.float32)
    dervative_o = tl.load(do + offset_block_m)
    query = tl.load(q + offset_block_m)
    if CAUSAL:
      # for the mask part of block_m
      dquery = _attention_bwd_dq(dquery, m, d, query, k, v, dervative_o, offset_batch_head,
                          offset_along_n, offset_along_h, block_m, block_n, batch_idx, head_idx, ctxid, num_heads, n_ctx, hidden_dim, sm_scale, causal=True, mask=True)
      # for the non-mask part for data before the mask (past data)
      dquery = _attention_bwd_dq(dquery, m, d, query, k, v, dervative_o, offset_batch_head,
                          offset_along_n, offset_along_h, block_m, block_n, batch_idx, head_idx, ctxid, num_heads, n_ctx, hidden_dim, sm_scale, causal=True, mask=False)
    else:
      # for non-causal, we feed both the past and future data together as there is no mask
      dquery = _attention_bwd_dq(dquery, m, d, query, k, v, dervative_o, offset_batch_head,
                          offset_along_n, offset_along_h, block_m,block_n, batch_idx, head_idx, ctxid, num_heads, n_ctx, hidden_dim, sm_scale, causal=False, mask=False)
    tl.store(dq + offset_block_m, dquery)
