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
  o_do = tl.sum(o*do, axis=1)
  delta = delta_ptr + batch_idx*heads*n_ctx + head_idx*n_ctx + offs_pre_block
  tl.store(delta, o_do)

@triton.autotune(
    configs=[
        triton.Config({'block_m':64, 'block_n':64}, num_warps=4, num_stages=1),
        triton.Config({'block_m':32,  'block_n':64}, num_warps=4, num_stages=1),
        triton.Config({'block_m':64, 'block_n':64}, num_warps=4, num_stages=1),
    ],
    key=['n_ctx', 'hidden_dim'],   # runtime-dependent shapes
)
@triton.jit
def _attention_bwd(q, k, v, do, dq, dk, dv, m, d,
                   sm_scale: tl.constexpr, batch: tl.constexpr, num_heads: tl.constexpr,
                   n_ctx: tl.constexpr, hidden_dim: tl.constexpr, bulk_slice_factor: tl.constexpr, block_m: tl.constexpr,
                   block_n: tl.constexpr):
  LN2 = 0.6931471824645996  # = ln(2)
  # current context block
  ctxid = tl.program_id(0)
  # current head
  hzid = tl.program_id(1)
  batch_idx = hzid // batch
  head_idx = hzid % batch
  # init offset can be used for both dkdv & dq
  init_offset = ctxid*block_m + head_idx*n_ctx + batch_idx*num_heads*n_ctx
  y_dim:tl.constexpr = batch * num_heads * n_ctx

  desc_v = tl.make_tensor_descriptor(v, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                        block_shape=[block_m, hidden_dim])
  desc_k = tl.make_tensor_descriptor(k, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                    block_shape=[block_m, hidden_dim])
  

  desc_dv = tl.make_tensor_descriptor(dv, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                        block_shape=[block_m, hidden_dim])
  desc_dk = tl.make_tensor_descriptor(dk, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                    block_shape=[block_m, hidden_dim])
  desc_dq = tl.make_tensor_descriptor(dq, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                        block_shape=[block_m, hidden_dim])
  
  dvalue = tl.zeros([block_m, hidden_dim], dtype=tl.float32)
  dkey = tl.zeros([block_m, hidden_dim], dtype=tl.float32)
  dquery = tl.zeros([block_m, hidden_dim], dtype=tl.float32)

  mask_block_n:tl.constexpr = block_n // bulk_slice_factor
  key = desc_k.load([init_offset,0])
  value = desc_v.load([init_offset,0])

  desc_query_mask_block_n = tl.make_tensor_descriptor(q, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                          block_shape=[mask_block_n, hidden_dim])
  desc_do_mask_block_n = tl.make_tensor_descriptor(do, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                    block_shape=[mask_block_n, hidden_dim])
  desc_key_mask_block_n = tl.make_tensor_descriptor(k, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                    block_shape=[mask_block_n, hidden_dim])
  offset_m = init_offset + tl.arange(0, block_m)
  offset = init_offset
  # Mask
  for sub_block_n in tl.range(0, block_m, mask_block_n):
    offset += sub_block_n
    offset_n = offset + tl.arange(0, mask_block_n)
    dquery_temp, dkey_temp, dvalue_temp = _attention_bwd_dqdkdv_per_sublock_n(dquery, dkey, dvalue, m, d,
                                      key, value, desc_query_mask_block_n,
                                      desc_do_mask_block_n, desc_key_mask_block_n, True, offset_m, offset_n, offset)
    dkey += dkey_temp
    dvalue += dvalue_temp
    dquery += dquery_temp

  # right of mask for non mask regions
  desc_query_block_n = tl.make_tensor_descriptor(q, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                          block_shape=[block_n, hidden_dim])
  desc_do_block_n = tl.make_tensor_descriptor(do, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                    block_shape=[block_n, hidden_dim])
  desc_key_block_n = tl.make_tensor_descriptor(k, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                    block_shape=[block_n, hidden_dim])

  diff: tl.constexpr = n_ctx-(ctxid*block_m + block_m)
  offset =  init_offset + block_m
  for sub_block_n in tl.range(0, diff, block_n):
    offset += sub_block_n
    offset_n = offset + tl.arange(0, block_n)
    dquery_temp, dkey_temp, dvalue_temp = _attention_bwd_dqdkdv_per_sublock_n(dquery, dkey, dvalue, m, d,
                                      key, value, desc_query_block_n,
                                      desc_do_block_n, desc_key_block_n, False, offset_m, offset_n, offset)
    dkey += dkey_temp
    dvalue += dvalue_temp
    dquery += dquery_temp

  desc_dv.store([init_offset, 0], dvalue)
  desc_dk.store([init_offset, 0], dkey*sm_scale)
  desc_dq.store([init_offset, 0], dquery*LN2)


@triton.jit
def _attention_bwd_dqdkdv_per_sublock_n(dquery, dkey, dvalue, m, d,
                                      key, value, desc_query_mask_block_n, desc_do_mask_block_n,
                                      desc_key_mask_block_n, MASK, offset_m, offset_n, offset):
  query = desc_query_mask_block_n.load([offset,0])
  key_n = desc_key_mask_block_n.load([offset,0])
  max_tensor = tl.load(m+offset_n)
  qkT = tl.dot(key, tl.trans(query))
  pT = tl.math.exp2(qkT - max_tensor[None, :])
  if MASK:
    mask = (offset_n[None, :] >= offset_m[:, None])
    pT = tl.where(mask, pT, 0.0)
  do = desc_do_mask_block_n.load([offset, 0]).to(tl.bfloat16)
  dvalue += tl.dot(pT.to(tl.bfloat16), do)
  delta = tl.load(d+offset_n)
  doT = tl.trans(do)
  dpT = tl.dot(value.to(tl.bfloat16), doT).to(tl.float32)
  dsT = pT * (dpT - delta[None, :])
  dsT = dsT.to(tl.bfloat16)
  dquery += tl.dot(dsT, key_n.to(tl.bfloat16))
  dkey += tl.dot(dsT, query.to(tl.bfloat16))
  return dquery, dkey, dvalue