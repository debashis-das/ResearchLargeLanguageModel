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