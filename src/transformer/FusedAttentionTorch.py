import torch

def _attention_forward_inner_non_mask(acc, l_i, m_i, q, desc_k, desc_v, 
                             offset_y, dtype, start_m, qk_scale,
                             block_m, hidden_dim, block_n,
                             offs_m, offs_n, n_ctx, wrap_specialize):
    lo, hi = 0, (start_m+1)*block_m
    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo
    for start_n in torch.arange(lo, hi, block_n):
        print(f"start_n : {start_n} : offset of k [{offsetk_y},0]")
        print("no mask used")
        print(f"Offset of v [0, {offsetv_y}]")
        offsetk_y += block_n
        offsetv_y += block_n

def _attention_forward_inner_mask(acc, l_i, m_i, q, desc_k, desc_v, 
                             offset_y, dtype, start_m, qk_scale,
                             block_m, hidden_dim, block_n, stage,
                             offs_m, offs_n, n_ctx, wrap_specialize):
    print(f"Stage : {stage}")
    if stage == 1:
        lo, hi = 0, start_m*block_m
    elif stage == 2:
        lo, hi = start_m*block_m, (start_m+1)*block_m
    else:
        print("Non-causal case")
        return
    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo
    for start_n in torch.arange(lo, hi, block_n):
        print(f"start_n : {start_n} : offset of k [{offsetk_y},0]")
        if stage == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            print(f"mask is used {mask}")
        else:
            print("no mask used")
        print(f"Offset of v [0, {offsetv_y}]")
        offsetk_y += block_n
        offsetv_y += block_n

def _attention_forward(sm_scale, max_tensor, num_heads, n_ctx, desc_q, desc_k, desc_v, desc_o, hidden_dim, block_m, block_n, mask_region, wrap_specialize):
    dtype = torch.bfloat16
    assert block_n <= hidden_dim
    for start_m in range(desc_q.shape[1]//block_m):
        for off_h in range(num_heads):
            print(f"start_m : {start_m}, off_h : {off_h}")
            # y_dim = num_heads*n_ctx
            offset_y = off_h*n_ctx
            print(f"offset_y : {offset_y}")
            qo_offset_y = offset_y + start_m*block_m
            print(f"qo_offset_y : {qo_offset_y}")
            offs_m = start_m*block_m + torch.arange(0, block_m)
            offs_n = torch.arange(0, block_n)
            print(f"offs_m : {offs_m}")
            print(f"offs_n : {offs_n}")
            # initialize pointer to m and l
            m_i = torch.zeros([block_m], dtype=torch.float32) - float("inf")
            l_i = torch.zeros([block_m], dtype=torch.float32) + 1.0
            acc = torch.zeros([block_m, hidden_dim], dtype=torch.float32)
            # load scales
            qk_scale = sm_scale
            qk_scale *= 1.44269504 #1/log(2)
            # q = q.load([qo_offset_y,0])
            print(f"q load : {[qo_offset_y, 0]}")
            # q = torch.randn([qo_offset_y, 0], dtype=dtype, device=device, requires_grad=True)
            if mask_region:
                _attention_forward_inner_mask(acc, l_i, m_i, desc_q, desc_k, desc_v, offset_y, dtype, start_m, qk_scale, block_m, hidden_dim, block_n, 1, offs_m, offs_n, n_ctx, wrap_specialize)
                _attention_forward_inner_mask(acc, l_i, m_i, desc_q, desc_k, desc_v, offset_y, dtype, start_m, qk_scale, block_m, hidden_dim, block_n, 2, offs_m, offs_n, n_ctx, wrap_specialize)
            else:
                _attention_forward_inner_non_mask(acc, l_i, m_i, desc_q, desc_k, desc_v, offset_y, dtype, start_m, qk_scale, block_m, hidden_dim, block_n, offs_m, offs_n, n_ctx, wrap_specialize)
             

# Assumption that it is used only for causal case
def attention(q, k, v, block_m, block_n, num_heads, sm_scale, device, q_index, kv_index, wrap_specialize=True):
    HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
    q_with_head = q.unsqueeze(0).expand(num_heads, -1, -1)
    k_with_head = k.unsqueeze(0).expand(num_heads, -1, -1)
    v_with_head = v.unsqueeze(0).expand(num_heads, -1, -1)
    o_with_head = torch.empty_like(q_with_head)
    print(q_with_head.shape, k_with_head.shape, v_with_head.shape)
    BLOCK_M = block_m
    BLOCK_N = block_n
    M = torch.empty(q_with_head.shape[0], q_with_head.shape[1], device=device, dtype=torch.float32)
    print(f"Max tokens : {M.shape}")
    grid = (q_with_head.shape[1]//BLOCK_M, num_heads, 1)
    print(f"Grid : {grid}")
    # Attention forward
    if q_index == kv_index:
        # mask region
        _attention_forward(sm_scale, M, num_heads, q_with_head.shape[1],
                        q_with_head, k_with_head, v_with_head, o_with_head,
                        q_with_head.shape[-1], BLOCK_M, BLOCK_N, True, wrap_specialize)
    else:
        # non-mask region
        _attention_forward(sm_scale, M, num_heads, q_with_head.shape[1],
                        q_with_head, k_with_head, v_with_head, o_with_head,
                        q_with_head.shape[-1], BLOCK_M, BLOCK_N, False, wrap_specialize)


def attention_bwd_pre_process(o_ptr, do_ptr, delta_ptr, n_ctx, pre_block, head, hidden):
    for pre_block_per_nctx in range(n_ctx//pre_block):
        for off_h in range(head):
            offs_pre_block = pre_block_per_nctx*pre_block + torch.arange(0, pre_block)
            offs_hid = torch.arange(0, hidden)
            offset = off_h*n_ctx + offs_pre_block[:,None]*hidden + offs_hid[None,:]
            print(f"O and do offset({pre_block_per_nctx}, {off_h}) : {offset}")
            # o = tl.load(o_ptr + offset)
            # do = tl.load(do_ptr + offset)
            # o_do = tl.sum(o*do, axis=1)
            print(f"O_do result offset({pre_block_per_nctx}, {off_h}) : {off_h*n_ctx + offs_pre_block}")
            # tl.store(delta+off_h*n_ctx + offs_pre_block,o_do)

def attention_bwd(q, k, v, block_m, block_n, num_heads, sm_scale, device, q_index, kv_index, wrap_specialize=True):
    o = torch.empty_like(q)
    do = torch.rand_like(o)

    print(q.shape, k.shape, v.shape)
    BLOCK_M = block_m
    BLOCK_N = block_n
    pre_block = 128
    num_hiddens = q.shape[-1]
    n_ctx = q.shape[1]
    grid = (q.shape[1]//pre_block, num_heads, 1)
    print(f"Grid : {grid}")
    delta = torch.empty((num_heads, n_ctx), dtype=torch.float32)
    # Preprocess
    attention_bwd_pre_process(o, do, delta, n_ctx, pre_block, num_heads, num_hiddens)
    # dq 
    





