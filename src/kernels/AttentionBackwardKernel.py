import triton
import triton.language as tl

@triton.jit
def _attention_bwd_pre_process(o_ptr, do_ptr, delta_ptr, n_ctx, pre_block, head, hidden):
    pre_block_per_nctx = tl.program_id(0)
    off_h = tl.program_id(1)
    offs_pre_block = pre_block_per_nctx*pre_block + tl.arange(0, pre_block)
    offs_hid = tl.arange(0, hidden)
    offset = off_h*n_ctx + offs_pre_block[:,None]*hidden + offs_hid[None,:]
    # print(f"O and do offset({pre_block_per_nctx}, {off_h}) : {offset}")
    o = tl.load(o_ptr + offset)
    do = tl.load(do_ptr + offset)
    o_do = tl.sum(o*do, axis=1)
    # print(f"O_do result offset({pre_block_per_nctx}, {off_h}) : {off_h*n_ctx + offs_pre_block}")
    tl.store(delta_ptr+off_h*n_ctx + offs_pre_block,o_do)