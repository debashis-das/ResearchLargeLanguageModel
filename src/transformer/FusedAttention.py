import torch
import gc
import torch.distributed as dist

# from kernels.AttentionForwardKernel import _attention_forward
# from kernels.AttentionBackwardKernel import _attention_bwd_pre_process, _attention_bwd

def identify_nodes_for_qkv(world_size:int):
    assert world_size % 2 == 0
    min_partitons_per_node = ((world_size * (world_size+1))//2)//world_size
    work_per_node = {}
    exe_order_per_rank_v = [[] for _ in range(world_size)]
    exe_order_per_rank_h = [[] for _ in range(world_size)]
    exe_order_per_rank_unaligned = [[] for _ in range(world_size)]
    for i in range(world_size):
        work_per_node[i] = min_partitons_per_node + (1 if i < world_size//2 else 0)
    for i in range(world_size):
        q_node_idx = i
        while work_per_node[i] > 0 and q_node_idx < world_size:
            exe_order_per_rank_v[i].append((q_node_idx,i))
            exe_order_per_rank_h[q_node_idx].append((q_node_idx, i))
            work_per_node[i] -= 1
            q_node_idx += 1
        dest_rank = world_size-1-i
        while q_node_idx < world_size:
            exe_order_per_rank_unaligned[dest_rank].append((q_node_idx,i))
            q_node_idx += 1
    return exe_order_per_rank_v, exe_order_per_rank_h, exe_order_per_rank_unaligned

def nodes_partion_q_fixed_kv(exe_order_per_rank_h, exe_order_per_rank_v, rank, q, k, v, grid, 
                             sm_scale, M, num_heads, n_ctx, 
                             hidden_dim, block_m, block_n, warp_specialize):
    recv_q = torch.empty_like(q)
    for (q_rank, dst_rank) in exe_order_per_rank_h[rank]:
        if q_rank != dst_rank:
            req_rec_q = dist.isend(q, dst=dst_rank)
            req_rec_q.wait()
            
    for (q_rank, kv_rank) in exe_order_per_rank_v[rank]:
        if kv_rank == rank:
            req_rec_q = dist.irecv(recv_q, src=q_rank)
            req_rec_q.wait()
            # print(f"Rec1(s:{q_rank},c:{rank}) {recv_q.shape}, {k.shape}, {v.shape} : {recv_q.stride()}, {k.stride()}, {v.stride()}")
            o = torch.ones_like(q)
            M = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
            sft_dem = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
            # _attention_forward[grid](sm_scale, M, sft_dem, num_heads, n_ctx,
            #             q, k, v, o,
            #             hidden_dim, block_m, block_n, True, warp_specialize)
            req_rec_o = dist.isend(o, dst=q_rank, tag=0)
            req_rec_m = dist.isend(M, dst=q_rank, tag=1)
            req_rec_sft_d = dist.isend(sft_dem, dst=q_rank, tag=2)
            req_rec_o.wait()
            req_rec_m.wait()
            req_rec_sft_d.wait()
            # gc.collect()
            # torch.cuda.empty_cache()

def nodes_partion_vary_qkv(exe_order_per_rank_unaligned, rank, q, k, v, grid, 
                           sm_scale, M, num_heads, n_ctx, 
                           hidden_dim, block_m, block_n, warp_specialize, current_o, current_m, current_sft_d):
    recv_k = torch.empty_like(k)
    recv_v = torch.empty_like(v)

    input_2_send = False
    input_2_recv = False
    for rank_in_list, list_per_rank in enumerate(exe_order_per_rank_unaligned):
      for (q_rank, kv_rank) in list_per_rank:
        if q_rank == rank and rank_in_list != rank:
          req_rec_q = dist.isend(q, dst=rank_in_list)
          req_rec_q.wait()
          # print(f"input send (send from:{rank},dst:{rank_in_list})")
        if kv_rank == rank and rank_in_list != rank and not input_2_send:
          req_rec_k = dist.isend(k, dst=rank_in_list)
          req_rec_v = dist.isend(v, dst=rank_in_list)
          req_rec_k.wait()
          req_rec_v.wait()
          # print(f"input2 send (send from:{rank},dst:{rank_in_list})")
          input_2_send = True
      
      if len(list_per_rank) != 0 and rank_in_list == rank :
        if not input_2_recv:
          req_rec_k = dist.irecv(recv_k, src=list_per_rank[0][1])
          req_rec_v = dist.irecv(recv_v, src=list_per_rank[0][1])
          req_rec_k.wait()
          req_rec_v.wait()
          # print(f"input2 irecv (src:{list_per_rank[0][1]},dest recevied to :{rank})")
          input_2_recv = True
        for (q_rank, kv_rank) in list_per_rank:
          recv_q = torch.empty_like(q)
          if q_rank == rank:
            recv_q = q
          else:
            req_rec_q = dist.irecv(recv_q, src=q_rank)
            req_rec_q.wait()
            # print(f"input irecv (src:{q_rank},dest recevied to :{rank})")
          print(f"R({rank}:{rank==q_rank})(s1:{q_rank},s2:{kv_rank}) {recv_q.shape}, {recv_k.shape}, {recv_v.shape}: {recv_q.stride()}, {recv_k.stride()}, {recv_v.stride()}")
          o = torch.ones_like(q)
          M = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
          sft_dem = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
          # _attention_forward[grid](sm_scale, M, num_heads, n_ctx,
          #               recv_q, recv_k, recv_v, o,
          #               hidden_dim, block_m, block_n, False, warp_specialize)
          if rank != q_rank:
            req_rec_o = dist.isend(o, dst=q_rank, tag=0)
            req_rec_m = dist.isend(M, dst=q_rank, tag=1)
            req_rec_sft_d = dist.isend(sft_dem, dst=q_rank, tag=2)
            req_rec_o.wait()
            req_rec_m.wait()
            req_rec_sft_d.wait()
          else:
            print(f"Current : {current_o.shape},{current_m.shape},{current_sft_d.shape}: {o.shape},{M.shape},{sft_dem.shape}")
            maximum = torch.maximum(current_m, M)
            scale_current = torch.exp2(current_m-maximum)
            scale_M = torch.exp2(M-maximum)
            current_sft_d = current_sft_d*scale_current+sft_dem*scale_M
            current_o = current_o*scale_current+o*scale_M
            current_m = maximum
            # gc.collect()
            # torch.cuda.empty_cache()
 

class _attention(torch.autograd.Function):
  
  # Assumption that it is used only for causal case
  @staticmethod
  def forward(ctx, q, k, v, block_m, block_n, num_heads, n_ctx, hidden_dim, sm_scale, world_size, rank, warp_specialize=True):
        if world_size != 1:
            exe_order_per_rank_v, exe_order_per_rank_h, exe_order_per_rank_unaligned  = identify_nodes_for_qkv(world_size)
        
        o = torch.empty_like(q)
        M = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
        sft_d = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)

        grid = (n_ctx//block_m, num_heads, 1)
        # print(f"Grid : {grid}")
        
        # Attention forward
        # mask region
        # _attention_forward[grid](sm_scale, M, sft_d, num_heads, n_ctx,
        #                 q, k, v, o,
        #                 hidden_dim, block_m, block_n, True, warp_specialize)
        # gc.collect()
        # torch.cuda.empty_cache()
        if world_size != 1:
            dist.barrier()
            exe_order_per_rank_v[rank].remove((rank,rank))
            exe_order_per_rank_h[rank].remove((rank,rank))
            dist.barrier()

            work_list_o = []
            work_list_m = []
            work_list_sft_d = []

            output_list_o = []
            output_list_m = []
            output_list_sft_d = []

            for _, kv_rank  in exe_order_per_rank_h[rank]:
              # print(f"Output({rank}) irecv src: {kv_rank}")
              recv_o = torch.empty_like(q)
              req_rec_o = dist.irecv(recv_o, src=kv_rank, tag=0)
              work_list_o.append(req_rec_o)
              output_list_o.append(recv_o)

              recv_m = torch.empty_like(M)
              req_rec_m = dist.irecv(recv_m, src=kv_rank, tag=1)
              work_list_m.append(req_rec_m)
              output_list_m.append(recv_m)

              recv_sft_d = torch.empty_like(sft_d)
              req_rec_sft_d = dist.irecv(recv_sft_d, src=kv_rank, tag=2)  
              work_list_sft_d.append(req_rec_sft_d)
              output_list_sft_d.append(recv_sft_d)

            for idx in range(world_size):
              for q_rank, kv_rank in exe_order_per_rank_unaligned[idx]:
                if rank == q_rank and idx != rank:
                  # print(f"Output({rank}) irecv src: {q_rank} : {kv_rank} : {idx}")
                  recv_o = torch.empty_like(q)
                  req_rec_o = dist.irecv(recv_o, src=q_rank, tag=0) 
                  work_list_o.append(req_rec_o)
                  output_list_o.append(recv_o)

                  recv_m = torch.empty_like(M)
                  req_rec_m = dist.irecv(recv_m, src=kv_rank, tag=1) 
                  work_list_m.append(req_rec_m)
                  output_list_m.append(recv_m)

                  recv_sft_d = torch.empty_like(sft_d)
                  req_rec_sft_d = dist.irecv(recv_sft_d, src=kv_rank, tag=2)  
                  work_list_sft_d.append(req_rec_sft_d)
                  output_list_sft_d.append(recv_sft_d)

            nodes_partion_q_fixed_kv(exe_order_per_rank_h, exe_order_per_rank_v,
                                     rank, q, k, v, grid, sm_scale, M, num_heads, n_ctx, 
                                     hidden_dim, block_m, block_n, warp_specialize)
            dist.barrier()
            nodes_partion_vary_qkv(exe_order_per_rank_unaligned, rank, q, k, v, 
                                   grid, sm_scale, M, num_heads, n_ctx, 
                                   hidden_dim, block_m, block_n, warp_specialize, o, M, sft_d)
            dist.barrier()
            for output_recv, work_o, m_recv, work_m, sft_d_recv, work_sft_d in zip(output_list_o, work_list_o, output_list_m, work_list_m, output_list_sft_d, work_list_sft_d):
              work_o.wait()
              work_m.wait()
              work_sft_d.wait()
              print(f"Output(R:{rank}) : {output_recv.shape}, {m_recv.shape}, {sft_d_recv.shape}")
              maximum = torch.maximum(M, m_recv)
              scale_current = torch.exp2(M-maximum)
              scale_recv = torch.exp2(m_recv-maximum)
              sft_d = sft_d*scale_current+sft_d_recv*scale_recv
              o = o*scale_current+output_recv*scale_recv
              M = maximum
            o = o / sft_d[:,None]

        ctx.save_for_backward(q,k,v,o,M)
        ctx.sm_scale = sm_scale
        ctx.hidden_dim = hidden_dim
        ctx.rank = rank
        ctx.num_heads = num_heads
        return o, M
  
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
      # _attention_bwd_pre_process[grid_preprocess](o, do, delta, n_ctx, pre_block, num_heads, num_hiddens)
      # bwd
      dq = torch.empty_like(q)
      dk = torch.empty_like(k)
      dv = torch.empty_like(v)
      bulk_slice_factor = 1
      grid_bwd = (n_ctx//block_m, num_heads, 1)
      print(f"Grid (bwd) : {grid_bwd}")
      gc.collect()
      torch.cuda.empty_cache()
      # _attention_bwd[grid_bwd](q, k, v, do, dq, dk, dv, M, delta, sm_scale, num_heads, n_ctx, 
      #                          num_hiddens, block_m, block_n, bulk_slice_factor)
      print(f"dv : {dv}")
      print(f"dk : {dk}")
      print(f"dq : {dq}")