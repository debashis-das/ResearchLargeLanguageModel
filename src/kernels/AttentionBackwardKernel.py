# import torch
import triton
import triton.language as tl

@triton.jit
def _attention_bwd_pre_process(o_ptr, do_ptr, delta_ptr, 
                               n_ctx: tl.constexpr,
                               pre_block: tl.constexpr,
                               heads: tl.constexpr,
                               hidden: tl.constexpr):
  pre_block_per_nctx = tl.program_id(0)
  off_h = tl.program_id(1)
  # for off_h in range(heads):
  #   for pre_block_per_nctx in range(n_ctx//pre_block):
  offs_pre_block = pre_block_per_nctx*pre_block + tl.arange(0, pre_block)
  offs_hid = tl.arange(0, hidden)
  offset = off_h*n_ctx*hidden + offs_pre_block[:,None]*hidden + offs_hid[None,:]
  # print(f"O and do offset({off_h}, {pre_block_per_nctx}) : {offset}")
  # o = torch.randn_like(offset, dtype=tl.bfloat16)
  # do = torch.rand_like(offset, dtype=tl.bfloat16)
  # o_od = torch.sum(o*do, 1)
  # print(f"O_do result offset({off_h}, {pre_block_per_nctx}) : {o_od.shape}")
  o = tl.load(o_ptr + offset)
  do = tl.load(do_ptr + offset)
  o_do = tl.sum(o*do, axis=1)
  # print(f"O_do result offset({off_h}, {pre_block_per_nctx}) : {off_h*n_ctx + offs_pre_block}")
  delta = delta_ptr + off_h*n_ctx + offs_pre_block
  tl.store(delta, o_do)


@triton.jit
def _attention_bwd(q, k, v, sm_scale, do, dq, dk, dv, m, d, num_heads, n_ctx, hidden_dim, block_m, block_n, bulk_slice_factor):
  LN2 = 0.6931471824645996  # = ln(2)
  # current context block 
  ctxid = tl.program_id(0)
  # current head
  hid = tl.program_id(1)
  # for mask we take half of the actual block_m value
  y_dim = num_heads * n_ctx
  init_offset = ctxid*block_m + hid*n_ctx
  
  desc_v = tl.make_tensor_descriptor(v, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                        block_shape=[block_n, hidden_dim])
  desc_k = tl.make_tensor_descriptor(k, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                    block_shape=[block_n, hidden_dim])
  
  # derivatives
  desc_dq = tl.make_tensor_descriptor(dq, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                        block_shape=[block_m, hidden_dim])
  
  desc_dv = tl.make_tensor_descriptor(dv, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                        block_shape=[block_n, hidden_dim])
  desc_dk = tl.make_tensor_descriptor(dk, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                    block_shape=[block_n, hidden_dim])
  dvalue = tl.zeros([block_m, hidden_dim], dtype=tl.float32)
  dkey = tl.zeros([block_m, hidden_dim], dtype=tl.float32)
  
  mask_block_n1 = block_n // bulk_slice_factor
  key = desc_k.load([init_offset,0])
  value = desc_v.load([init_offset,0])

  desc_query_mask_block_n1 = tl.make_tensor_descriptor(q, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                          block_shape=[mask_block_n1, hidden_dim])
  desc_queryT_mask_block_n1 = tl.make_tensor_descriptor(q, shape=[hidden_dim, y_dim], strides=[1, hidden_dim],
                                          block_shape=[hidden_dim, mask_block_n1])
  desc_do_mask_block_n1 = tl.make_tensor_descriptor(do, shape=[y_dim, hidden_dim], strides=[hidden_dim, 1],
                                    block_shape=[mask_block_n1, hidden_dim])
  desc_doT_mask_block_n1 = tl.make_tensor_descriptor(do, shape=[hidden_dim, y_dim], strides=[1, hidden_dim],
                                    block_shape=[hidden_dim, mask_block_n1])
  start_m = init_offset + tl.arange(0, mask_block_n1)
  start_n = start_m
  current_m = start_m

  # Mask
  for block_m_sub_block_n in tl.range(0, block_m, mask_block_n1):
    offset =  block_m_sub_block_n*mask_block_n1 + init_offset
    _attention_bwd_dkdv_per_sublock_n(dkey, dvalue, m, d, 
                                      key, value, desc_query_mask_block_n1, desc_queryT_mask_block_n1, 
                                      desc_do_mask_block_n1, desc_doT_mask_block_n1, True, current_m, start_n, offset)
    current_m += mask_block_n1

  # left of Mask  
  for block_n_idx in tl.range(block_m, n_ctx, block_n):
    offset =  block_m + block_n_idx*block_n + init_offset
    _attention_bwd_dkdv_per_sublock_n(dkey, dvalue, m, d, 
                                      key, value, desc_query_mask_block_n1, desc_queryT_mask_block_n1, 
                                      desc_do_mask_block_n1, desc_doT_mask_block_n1, False, current_m, start_n, offset)
    current_m += block_n

  desc_dv.store([init_offset, 0], dvalue)
  desc_dk.store([init_offset, 0], dkey*sm_scale)
  
  desc_vT = tl.make_tensor_descriptor(v, shape=[hidden_dim, y_dim], strides=[1, hidden_dim],
                                        block_shape=[hidden_dim, block_n])
  desc_kT = tl.make_tensor_descriptor(k, shape=[hidden_dim, y_dim], strides=[1, hidden_dim],
                                    block_shape=[hidden_dim, block_n])
  dquery = tl.zeros([block_m, hidden_dim], dtype=tl.float32)
  query = desc_query_mask_block_n1.load([init_offset,0])
  max_offset = init_offset + tl.arange(0, block_m)
  max_tensor = tl.load(m+max_offset)
  # Mask
  for block_m_sub_block_n in tl.range(0, block_m, mask_block_n1):
    offset =  block_m_sub_block_n*mask_block_n1 + init_offset

def _attention_bwd_dq_per_sublock_n(query, desc_vT, desc_kT, max_tensor, start_m, dquery, offset):
  vT = tl.load(desc_vT)
  kT = tl.load(desc_kT)
  qk = tl.dot(query, kT)
  
  p = tl.math.exp2(qk - max_tensor)
  
  

def _attention_bwd_dkdv_per_sublock_n(dkey, dvalue, m, d, 
                                      key, value, desc_query_mask_block_n1, desc_queryT_mask_block_n1, desc_do_mask_block_n1,
                                      desc_doT_mask_block_n1, MASK, start_m, start_n, offset):
  queryT = desc_queryT_mask_block_n1.load([0, offset])
  max_tensor = tl.load(m+start_m)

  qkT = tl.dot(key, queryT)
  pT = tl.math.exp2(qkT - max_tensor[None, :])
  if MASK:
    mask = (start_m[None, :] > start_n[:, None])
    pT = tl.where(mask, pT, 0.0)
  do = desc_do_mask_block_n1.load([offset, 0])
  dvalue += tl.dot(pT.to(tl.bfloat16), do)
  delta = tl.load(d+start_m)
  doT = desc_doT_mask_block_n1.load([0, offset])
  dpT = tl.dot(value, doT).to(tl.float32)
  dsT = pT * (dpT - delta[None,:])
  dsT = dsT.to(tl.bfloat16)
  query = desc_query_mask_block_n1.load([offset, 0])
  dkey += tl.dot(dsT, query)
   


    





    



  
  
  

  