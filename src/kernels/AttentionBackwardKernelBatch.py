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
    o = tl.maximum(tl.minimum(o, 1.0e8), -1.0e8)
    do = tl.maximum(tl.minimum(do, 1.0e8), -1.0e8)
    o_do = tl.sum(o*do, axis=1)
    delta = delta_ptr + batch_idx*heads*n_ctx + head_idx*n_ctx + offs_pre_block
    tl.store(delta, o_do)

@triton.autotune(
    configs=[
        triton.Config({'block_m':32, 'block_n':32}, num_warps=4, num_stages=1),
        # triton.Config({'block_m':16, 'block_n':32}, num_warps=4, num_stages=1),
        triton.Config({'block_m':32, 'block_n':16}, num_warps=4, num_stages=1),
        triton.Config({'block_m':16, 'block_n':16}, num_warps=4, num_stages=1)
    ],
    key=['n_ctx', 'hidden_dim'],   # runtime-dependent shapes
)
@triton.jit
def _attention_bwd(q, k, v, do, dq, dk, dv, m, d, sft_d,
                   sm_scale: tl.constexpr, batch: tl.constexpr, num_heads: tl.constexpr,
                   n_ctx: tl.constexpr, hidden_dim: tl.constexpr, bulk_slice_factor: tl.constexpr, block_m: tl.constexpr,
                   block_n: tl.constexpr):
    # LN2 = 0.6931471824645996  # = ln(2)
    # current context block
    ctxid = tl.program_id(0)
    # current head
    hzid = tl.program_id(1)
    batch_idx = hzid // batch
    head_idx = hzid % batch
    # init offset can be used for both dkdv & dq
    init_offset = ctxid*block_m*hidden_dim + head_idx*n_ctx*hidden_dim + batch_idx*num_heads*n_ctx*hidden_dim
    y_dim:tl.constexpr = batch * num_heads * n_ctx * hidden_dim

    dvalue = tl.zeros([block_m, hidden_dim], dtype=tl.float32)
    dkey = tl.zeros([block_m, hidden_dim], dtype=tl.float32)
    dquery = tl.zeros([block_m, hidden_dim], dtype=tl.float32)
    offset_m = init_offset + tl.arange(0, block_m)

    row_offset_block_m = tl.arange(0, block_m)*hidden_dim
    col_offset_block_m = tl.arange(0, hidden_dim)
    offset_block_m = init_offset + row_offset_block_m[:, None] + col_offset_block_m[None, :]

    # pre-load max and softmax denominator for the current block to L1
    offset_max_and_sft_denominator = ctxid*block_m + head_idx*n_ctx + batch_idx*num_heads*n_ctx
    row_offset_block_m_of_max_sft_denominator_delta = offset_max_and_sft_denominator + tl.arange(0, block_m)
    max_tensor = tl.load(m + row_offset_block_m_of_max_sft_denominator_delta)
    softmax_denominator = tl.load(sft_d + row_offset_block_m_of_max_sft_denominator_delta)
    delta = tl.load(d+row_offset_block_m_of_max_sft_denominator_delta)
    # left of mask for non mask regions
    start = 0
    end = ctxid*block_m
    increment = block_n
    row_offset_block_n = tl.arange(0,hidden_dim)*n_ctx
    col_offset_block_n = tl.arange(0, block_n)
    row_offset_block_n_inverted = tl.arange(0, block_n)*hidden_dim
    col_offset_block_n_inverted = tl.arange(0, hidden_dim)
    query = tl.load(q + offset_block_m)  # pre-load to L1
    value = tl.load(v + offset_block_m)  # pre-load to L1
    MASK = False
    offset = ctxid*block_m + head_idx*n_ctx*hidden_dim + batch_idx*num_heads*n_ctx*hidden_dim
    offset_inverted = ctxid*block_m*hidden_dim + head_idx*n_ctx*hidden_dim + batch_idx*num_heads*n_ctx*hidden_dim
    for sub_block_n in tl.range(start, end, increment):
      offset += sub_block_n
      offset_n = offset + tl.arange(0, block_n)
      offset_block_n = offset + row_offset_block_n[:, None] + col_offset_block_n[None, :]
      offset_inverted += sub_block_n*hidden_dim
      offset_block_n_inverted = offset_inverted + row_offset_block_n_inverted[:, None] + col_offset_block_n_inverted[None, :]
      qkT = tl.dot(query, tl.load(k + offset_block_n))*sm_scale
      if MASK:
        mask = (offset_m[:, None] >= offset_n[None, :])
        qkT += tl.where(mask, 0, -1.0e8)
      p = tl.math.exp(qkT - max_tensor[:, None]) / softmax_denominator[:, None]
      dvalue += tl.dot(p,  tl.load(do + offset_block_n_inverted)).to(tl.float32)
      dp = tl.dot(value.to(tl.float64), tl.load(do + offset_block_n).to(tl.float64)).to(tl.float32)
      ds = p * (dp - delta[:, None])
      dquery += tl.dot(ds, tl.load(k + offset_block_n_inverted).to(tl.float32))
      dkey += tl.dot(ds, tl.load(q + offset_block_n_inverted).to(tl.float32))

    # mask regions
    mask_block_n:tl.constexpr = block_n // bulk_slice_factor
    # mask_block_n:tl.constexpr = block_n
    start = 0
    end = block_m
    increment = mask_block_n
    row_offset_block_n = tl.arange(0,hidden_dim)*n_ctx
    col_offset_block_n = tl.arange(0, mask_block_n)
    row_offset_block_n_inverted = tl.arange(0, mask_block_n)*hidden_dim
    col_offset_block_n_inverted = tl.arange(0, hidden_dim)
    MASK = True
    offset = ctxid*block_m + head_idx*n_ctx*hidden_dim + batch_idx*num_heads*n_ctx*hidden_dim
    offset_inverted = ctxid*block_m*hidden_dim + head_idx*n_ctx*hidden_dim + batch_idx*num_heads*n_ctx*hidden_dim
    for sub_block_n in tl.range(start, end, increment):
      offset += sub_block_n
      offset_n = offset + tl.arange(0, mask_block_n)
      offset_block_n = offset + row_offset_block_n[:, None] + col_offset_block_n[None, :]
      offset_inverted += sub_block_n*hidden_dim
      offset_block_n_inverted = offset_inverted + row_offset_block_n_inverted[:, None] + col_offset_block_n_inverted[None, :]
      qkT = tl.dot(query, tl.load(k + offset_block_n))*sm_scale
      if MASK:
        mask = (offset_m[:, None] >= offset_n[None, :])
        qkT = qkT + tl.where(mask, 0, -1.0e6)
      p = tl.math.exp(qkT - max_tensor[:, None]) / softmax_denominator[:, None]
      dvalue += tl.dot(p,  tl.load(do + offset_block_n_inverted)).to(tl.float32)
      dp = tl.dot(value.to(tl.float64), tl.load(do + offset_block_n).to(tl.float64)).to(tl.float32)
      ds = p * (dp - delta[:, None])
      dquery += tl.dot(ds, tl.load(k + offset_block_n_inverted).to(tl.float32))
      dkey += tl.dot(ds, tl.load(q + offset_block_n_inverted).to(tl.float32))

    tl.store(dv + offset_block_m, dvalue)
    tl.store(dk + offset_block_m, dkey)
    tl.store(dq + offset_block_m, dquery)

# @triton.jit
# def _attention_bwd_dqdkdv_per_sublock_n(dquery, dkey, dvalue, m, sft_d, d,
#                                       key, value, desc_query_mask_block_n, desc_do_mask_block_n,
#                                       desc_key_mask_block_n,
#                                         MASK, offset_m, offset_n, offset, sm_scale):
#   query = desc_query_mask_block_n.load([offset,0])
#   key_n = desc_key_mask_block_n.load([offset,0])
#   max_tensor = tl.load(m+offset_n)
#   softmax_denominator = tl.load(sft_d+offset_n)
#   kqT = tl.dot(key, tl.trans(query))*sm_scale
#   if MASK:
#     mask = (offset_n[None, :] >= offset_m[:, None])
#     kqT += tl.where(mask, 0, -1.0e8)
#   pT = tl.math.exp(kqT - max_tensor[None, :]) / softmax_denominator[None, :]
#   do = desc_do_mask_block_n.load([offset, 0]).to(tl.float32)
#   dvalue += tl.dot(pT, do).to(tl.float32)
#   delta = tl.load(d+offset_n)
#   doT = tl.trans(do)
#   dpT = tl.dot(value, doT).to(tl.float32)
#   dsT = pT * (dpT - delta[None, :])
#   dsT = dsT.to(tl.float32)
#   dquery += tl.dot(dsT, key_n.to(tl.float32))
#   dkey += tl.dot(dsT, query.to(tl.float32))
#   return dquery, dkey, dvalue