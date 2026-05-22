import sys
import threading
import torch
import gc
import torch.distributed as dist
import triton
import logging

from kernels.AttentionForwardKernelBatch import _attention_forward
from kernels.AttentionBackwardKernelBatch import _attention_bwd, _attention_bwd_pre_process
from kernels.AttentionBackwardKernelBatchFromDoc import _attn_bwd, _attn_bwd_preprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
semaphore = threading.Semaphore()

# def identify_nodes_for_qkv(world_size:int):
#     assert world_size % 2 == 0
#     min_partitons_per_node = ((world_size * (world_size+1))//2)//world_size
#     work_per_node = {}
#     exe_order_per_rank_v = [[] for _ in range(world_size)]
#     exe_order_per_rank_h = [[] for _ in range(world_size)]
#     exe_order_per_rank_unaligned = [[] for _ in range(world_size)]
#     for i in range(world_size):
#         work_per_node[i] = min_partitons_per_node + (1 if i < world_size//2 else 0)
#     for i in range(world_size):
#         q_node_idx = i
#         while work_per_node[i] > 0 and q_node_idx < world_size:
#             exe_order_per_rank_v[i].append((q_node_idx,i))
#             exe_order_per_rank_h[q_node_idx].append((q_node_idx, i))
#             work_per_node[i] -= 1
#             q_node_idx += 1
#         dest_rank = world_size-1-i
#         while q_node_idx < world_size:
#             exe_order_per_rank_unaligned[dest_rank].append((q_node_idx,i))
#             q_node_idx += 1
#     return exe_order_per_rank_v, exe_order_per_rank_h, exe_order_per_rank_unaligned

# def nodes_partion_q_fixed_kv_forward(exe_order_per_rank_h, exe_order_per_rank_v, rank, q, k, v, grid, 
#                              sm_scale, M, batch, num_heads, n_ctx, 
#                              hidden_dim, warp_specialize):
#     wait_list = []
#     for (q_rank, dst_rank) in exe_order_per_rank_h[rank]:
#         if q_rank != dst_rank:
#             req_rec_q = dist.isend(q, dst=dst_rank)
#             wait_list.append(req_rec_q)

#     for work in wait_list:
#        work.wait() 
            
#     for (q_rank, kv_rank) in exe_order_per_rank_v[rank]:
#         if kv_rank == rank:
#             recv_q = torch.empty_like(q)
#             req_rec_q = dist.irecv(recv_q, src=q_rank)
#             req_rec_q.wait()
#             # logging.debug(f"Rec1_forward(s:{q_rank},c:{rank}) {recv_q.shape}, {k.shape}, {v.shape} : {recv_q.stride()}, {k.stride()}, {v.stride()}")
#             with semaphore:
#               o = torch.ones_like(q)
#               M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.bfloat16)
#               sft_dem = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.bfloat16)

#               # _attention_forward[grid](sm_scale, M, sft_dem, batch, num_heads, n_ctx,
#               #             recv_q, k, v, o,
#               #             hidden_dim, True, warp_specialize)
#               req_rec_o = dist.isend(o, dst=q_rank, tag=0)
#               req_rec_m = dist.isend(M, dst=q_rank, tag=1)
#               req_rec_sft_d = dist.isend(sft_dem, dst=q_rank, tag=2)
#               req_rec_o.wait()
#               req_rec_m.wait()
#               req_rec_sft_d.wait()
#               gc.collect()
#               torch.cuda.empty_cache()

# def nodes_partion_vary_qkv_forward(exe_order_per_rank_unaligned, rank, q, k, v, grid, 
#                            sm_scale, M, batch, num_heads, n_ctx, 
#                            hidden_dim, current_o, current_m, current_sft_d, warp_specialize):
#     input_2_send = False
#     input_2_recv = False
#     for rank_in_list, list_per_rank in enumerate(exe_order_per_rank_unaligned):
#       for (q_rank, kv_rank) in list_per_rank:
#         if q_rank == rank and rank_in_list != rank:
#           req_rec_q = dist.isend(q, dst=rank_in_list)
#           req_rec_q.wait()
#           # logging.debug(f"input send (send from:{rank},dst:{rank_in_list})")
#         if kv_rank == rank and rank_in_list != rank and not input_2_send:
#           req_rec_k = dist.isend(k, dst=rank_in_list)
#           req_rec_v = dist.isend(v, dst=rank_in_list)
#           req_rec_k.wait()
#           req_rec_v.wait()
#           # logging.debug(f"input2 send (send from:{rank},dst:{rank_in_list})")
#           input_2_send = True
      
#       if len(list_per_rank) != 0 and rank_in_list == rank :
#         recv_k = torch.empty_like(k)
#         recv_v = torch.empty_like(v)
#         if not input_2_recv:
#           req_rec_k = dist.irecv(recv_k, src=list_per_rank[0][1])
#           req_rec_v = dist.irecv(recv_v, src=list_per_rank[0][1])
#           req_rec_k.wait()
#           req_rec_v.wait()
#           # logging.debug(f"input2 irecv (src:{list_per_rank[0][1]},dest recevied to :{rank})")
#           input_2_recv = True
#         for (q_rank, kv_rank) in list_per_rank:
#           recv_q = torch.empty_like(q)
#           if q_rank == rank:
#             recv_q = q
#           else:
#             req_rec_q = dist.irecv(recv_q, src=q_rank)
#             req_rec_q.wait()
#           #   logging.debug(f"input irecv (src:{q_rank},dest recevied to :{rank})")
#           # logging.debug(f"R({rank}:{rank==q_rank})(s1:{q_rank},s2:{kv_rank}) {recv_q.shape}, {recv_k.shape}, {recv_v.shape}: {recv_q.stride()}, {recv_k.stride()}, {recv_v.stride()}")
#           o = torch.ones_like(q)
#           M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.bfloat16)
#           sft_dem = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.bfloat16)
#           with semaphore:
#             # _attention_forward[grid](sm_scale, M, batch, num_heads, n_ctx,
#             #               recv_q, recv_k, recv_v, o,
#             #               hidden_dim, False, warp_specialize)
#             if rank != q_rank:
#               req_rec_o = dist.isend(o, dst=q_rank, tag=0)
#               req_rec_m = dist.isend(M, dst=q_rank, tag=1)
#               req_rec_sft_d = dist.isend(sft_dem, dst=q_rank, tag=2)
#               req_rec_o.wait()
#               req_rec_m.wait()
#               req_rec_sft_d.wait()
#             else:
#               # logging.debug(f"Current : {current_o.shape},{current_m.shape},{current_sft_d.shape}: {o.shape},{M.shape},{sft_dem.shape}")
#               maximum = torch.maximum(current_m, M)
#               scale_current = torch.exp2(current_m-maximum)
#               scale_M = torch.exp2(M-maximum)
#               current_sft_d = current_sft_d*scale_current+sft_dem*scale_M
#               current_o = current_o*scale_current.unsqueeze(-1)+o*scale_M.unsqueeze(-1)
#               current_m = maximum
#               gc.collect()
#               torch.cuda.empty_cache()


# def nodes_partion_q_fixed_kv_backward(exe_order_per_rank_h, exe_order_per_rank_v, rank, q, k, v, grid_preprocess, grid_bwd,
#                                       o, do, M, pre_block, sm_scale, batch, num_heads, n_ctx, num_hiddens, 
#                                       bulk_slice_factor, warp_specialize):
#     recv_q = torch.empty_like(q)
#     wait_list = []
#     for (q_rank, dst_rank) in exe_order_per_rank_h[rank]:
#         if q_rank != dst_rank:
#             req_rec_q = dist.isend(q, dst=dst_rank)
#             wait_list.append(req_rec_q)

#     for work in wait_list:
#        work.wait() 
            
#     for (q_rank, kv_rank) in exe_order_per_rank_v[rank]:
#         if kv_rank == rank:
#             req_rec_q = dist.irecv(recv_q, src=q_rank)
#             req_rec_q.wait()
#             # logging.debug(f"Rec1_backward(s:{q_rank},c:{rank}) {recv_q.shape}, {k.shape}, {v.shape} : {recv_q.stride()}, {k.stride()}, {v.stride()}")
#             with semaphore:
#               delta = torch.empty((recv_q.shape[0], recv_q.shape[1]), device=recv_q.device, dtype=torch.bfloat16)
#               # _attention_bwd_pre_process[grid_preprocess](o, do, delta, batch, n_ctx, pre_block, num_heads, num_hiddens)
#               dq = torch.empty_like(recv_q)
#               dk = torch.empty_like(k)
#               dv = torch.empty_like(v)
#               # _attention_bwd[grid_bwd](recv_q, k, v, do, dq, dk, dv, M, delta, sm_scale, batch, num_heads, n_ctx, 
#               #                                num_hiddens, bulk_slice_factor,)
#               req_rec_dq = dist.isend(dq, dst=q_rank, tag=0)
#               req_rec_dk = dist.isend(dk, dst=q_rank, tag=1)
#               req_rec_dv = dist.isend(dv, dst=q_rank, tag=2)
#               req_rec_dq.wait()
#               req_rec_dk.wait()
#               req_rec_dv.wait()
#               gc.collect()
#               torch.cuda.empty_cache()
    

# def nodes_partion_vary_qkv_backward(exe_order_per_rank_unaligned, rank, q, k, v, grid_preprocess, grid_bwd,
#                                    o, do, M, current_dq, current_dk, current_dv, pre_block, sm_scale, batch, num_heads, n_ctx, num_hiddens, 
#                                    bulk_slice_factor, warp_specialize):
#     input_2_send = False
#     input_2_recv = False
#     for rank_in_list, list_per_rank in enumerate(exe_order_per_rank_unaligned):
#       for (q_rank, kv_rank) in list_per_rank:
#         if q_rank == rank and rank_in_list != rank:
#           req_rec_q = dist.isend(q, dst=rank_in_list)
#           req_rec_q.wait()
#           # logging.debug(f"input send (send from:{rank},dst:{rank_in_list})")
#         if kv_rank == rank and rank_in_list != rank and not input_2_send:
#           req_rec_k = dist.isend(k, dst=rank_in_list)
#           req_rec_v = dist.isend(v, dst=rank_in_list)
#           req_rec_k.wait()
#           req_rec_v.wait()
#           # logging.debug(f"input2 send (send from:{rank},dst:{rank_in_list})")
#           input_2_send = True
      
#       if len(list_per_rank) != 0 and rank_in_list == rank :
#         recv_k = torch.empty_like(k)
#         recv_v = torch.empty_like(v)
#         if not input_2_recv:
#           req_rec_k = dist.irecv(recv_k, src=list_per_rank[0][1])
#           req_rec_v = dist.irecv(recv_v, src=list_per_rank[0][1])
#           req_rec_k.wait()
#           req_rec_v.wait()
#           # logging.debug(f"input2 irecv (src:{list_per_rank[0][1]},dest recevied to :{rank})")
#           input_2_recv = True
#         for (q_rank, kv_rank) in list_per_rank:
#           recv_q = torch.empty_like(q)
#           if q_rank == rank:
#             recv_q = q
#           else:
#             req_rec_q = dist.irecv(recv_q, src=q_rank)
#             req_rec_q.wait()
#           #   logging.debug(f"input irecv (src:{q_rank},dest recevied to :{rank})")
#           # logging.debug(f"R({rank}:{rank==q_rank})(s1:{q_rank},s2:{kv_rank}) {recv_q.shape}, {recv_k.shape}, {recv_v.shape}: {recv_q.stride()}, {recv_k.stride()}, {recv_v.stride()}")
#           with semaphore:
#             delta = torch.empty((recv_q.shape[0], recv_q.shape[1]), device=recv_q.device, dtype=torch.bfloat16)
#             # _attention_bwd_pre_process[grid_preprocess](o, do, delta, batch, n_ctx, pre_block, num_heads, num_hiddens)
#             dq = torch.empty_like(recv_q)
#             dk = torch.empty_like(recv_k)
#             dv = torch.empty_like(recv_v)
#             # _attention_bwd[grid_bwd](req_rec_q, recv_k, recv_v, do, dq, dk, dv, M, delta, sm_scale, batch, num_heads, n_ctx, 
#             #                                 num_hiddens, bulk_slice_factor,)
#             if rank != q_rank:
#               req_rec_dq = dist.isend(dq, dst=q_rank, tag=0)
#               req_rec_dk = dist.isend(dk, dst=q_rank, tag=1)
#               req_rec_dv = dist.isend(dv, dst=q_rank, tag=2)
#               req_rec_dq.wait()
#               req_rec_dv.wait()
#               req_rec_dv.wait()
#               gc.collect()
#               torch.cuda.empty_cache()
#             else:
#               # print(f"Current : {current_dq.shape},{current_dk.shape},{current_dv.shape}: {dq.shape},{dv.shape},{dv.shape}")
#               current_dq += dq
#               current_dk += dk
#               current_dv += dv
#               gc.collect()
#               torch.cuda.empty_cache()

class _attention(torch.autograd.Function):
  
  # Assumption that it is used only for causal case
  @staticmethod
  def forward(ctx, q, k, v, world_size, rank, warp_specialize=True):
        if world_size != 1:
            exe_order_per_rank_v, exe_order_per_rank_h, exe_order_per_rank_unaligned  = identify_nodes_for_qkv(world_size)
        batch, num_heads, n_ctx, hidden_dim = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
        sm_scale = 1.0 / (hidden_dim ** 0.5)
        o = torch.zeros_like(q)
        M = torch.zeros((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.bfloat16)
        sft_d = torch.zeros((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.bfloat16)
        
        grid_fwd = lambda META: (
            triton.cdiv(n_ctx, META['block_m']),
            num_heads * batch,
            1
        )
        if q.dtype == torch.float16:
          lower_precision = True
        else:
          lower_precision = False
        # grid = (n_ctx//block_m, num_heads*batch, 1)
        # print(f"Grid : {grid}")
        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)
        # Attention forward
        # mask region
        _attention_forward[grid_fwd](sm_scale, M, sft_d, batch, num_heads, n_ctx,
                        q, k, v, o, lower_precision,
                        hidden_dim, True, warp_specialize,)
        # gc.collect()
        # torch.cuda.empty_cache()
        # if world_size != 1:
        #     dist.barrier()
        #     exe_order_per_rank_v[rank].remove((rank,rank))
        #     exe_order_per_rank_h[rank].remove((rank,rank))
        #     dist.barrier()

        #     work_list_o = []
        #     work_list_m = []
        #     work_list_sft_d = []

        #     output_list_o = []
        #     output_list_m = []
        #     output_list_sft_d = []

        #     for _, kv_rank  in exe_order_per_rank_h[rank]:
        #       # print(f"Output({rank}) irecv src: {kv_rank}")
        #       recv_o = torch.empty_like(q)
        #       req_rec_o = dist.irecv(recv_o, src=kv_rank, tag=0)
        #       work_list_o.append(req_rec_o)
        #       output_list_o.append(recv_o)

        #       recv_m = torch.empty_like(M)
        #       req_rec_m = dist.irecv(recv_m, src=kv_rank, tag=1)
        #       work_list_m.append(req_rec_m)
        #       output_list_m.append(recv_m)

        #       recv_sft_d = torch.empty_like(sft_d)
        #       req_rec_sft_d = dist.irecv(recv_sft_d, src=kv_rank, tag=2)  
        #       work_list_sft_d.append(req_rec_sft_d)
        #       output_list_sft_d.append(recv_sft_d)

        #     for idx in range(world_size):
        #       for q_rank, kv_rank in exe_order_per_rank_unaligned[idx]:
        #         if rank == q_rank and idx != rank:
        #           # print(f"Output({rank}) irecv src: {q_rank} : {kv_rank} : {idx}")
        #           recv_o = torch.empty_like(q)
        #           req_rec_o = dist.irecv(recv_o, src=q_rank, tag=0) 
        #           work_list_o.append(req_rec_o)
        #           output_list_o.append(recv_o)

        #           recv_m = torch.empty_like(M)
        #           req_rec_m = dist.irecv(recv_m, src=kv_rank, tag=1) 
        #           work_list_m.append(req_rec_m)
        #           output_list_m.append(recv_m)

        #           recv_sft_d = torch.empty_like(sft_d)
        #           req_rec_sft_d = dist.irecv(recv_sft_d, src=kv_rank, tag=2)  
        #           work_list_sft_d.append(req_rec_sft_d)
        #           output_list_sft_d.append(recv_sft_d)

        #     nodes_partion_q_fixed_kv_forward(exe_order_per_rank_h, exe_order_per_rank_v,
        #                              rank, q, k, v, grid_fwd, sm_scale, M, batch, num_heads, n_ctx, 
        #                              hidden_dim, warp_specialize)
        #     # dist.barrier()
        #     nodes_partion_vary_qkv_forward(exe_order_per_rank_unaligned, rank, q, k, v, 
        #                            grid_fwd, sm_scale, M, batch, num_heads, n_ctx, 
        #                            hidden_dim, o, M, sft_d, warp_specialize)
            
        #     for output_recv, work_o, m_recv, work_m, sft_d_recv, work_sft_d in zip(output_list_o, work_list_o, output_list_m, work_list_m, output_list_sft_d, work_list_sft_d):
        #       work_o.wait()
        #       work_m.wait()
        #       work_sft_d.wait()
        #       # logging.debug(f"Output(R:{rank}) : {output_recv.shape}, {m_recv.shape}, {sft_d_recv.shape}")
        #       with semaphore:
        #         maximum = torch.maximum(M, m_recv)
        #         scale_current = torch.exp2(M-maximum)
        #         scale_recv = torch.exp2(m_recv-maximum)
        #         sft_d = sft_d*scale_current+sft_d_recv*scale_recv
        #         o = o*scale_current.unsqueeze(-1)+output_recv*scale_recv.unsqueeze(-1)
        #         M = maximum
        #     o = o / sft_d.unsqueeze(-1)
        dist.barrier()
        ctx.save_for_backward(q,k,v,o,M)
        ctx.sm_scale = sm_scale
        ctx.hidden_dim = hidden_dim
        # ctx.rank = rank
        # ctx.world_size = world_size
        ctx.num_heads = num_heads
        ctx.batch = batch
        # print(o)
        return o

  @staticmethod
  def backward(ctx, do):
      q, k, v, o, M = ctx.saved_tensors
      sm_scale = ctx.sm_scale
      HEAD_DIM = ctx.hidden_dim
      # rank = ctx.rank
      # world_size = ctx.world_size
      N_HEAD = ctx.num_heads
      BATCH = ctx.batch
      # pre_block = 64
      # num_hiddens = q.shape[-1]
      N_CTX = q.shape[2]
      PRE_BLOCK = 128
      # print(f"Q shape : {q.shape} : N_CTX : {N_CTX}, Pre_block : {PRE_BLOCK}")
      assert N_CTX % PRE_BLOCK == 0
      delta = torch.empty_like(M, device=q.device, dtype=torch.float32)
      grid_preprocess = (N_CTX//PRE_BLOCK, N_HEAD*BATCH, 1)
      # Preprocess
      _attention_bwd_pre_process[grid_preprocess](
          o, do,
          delta, 
          BATCH, N_CTX, PRE_BLOCK, N_HEAD, HEAD_DIM)
      # logging.debug(f"[Attention] delta_doc({delta_doc.shape}): {torch.isnan(delta_doc).any()} : delta_doc.max(): {delta_doc.abs().max()} : delta_doc.min(): {delta_doc.abs().min()}")
      # logging.debug(f"[Attention] delta({delta.shape}): {torch.isnan(delta).any()} : delta.max(): {delta.abs().max()} : delta.min(): {delta.abs().min()}")
      # bwd
      # dq = torch.zeros_like(q)
      # dk = torch.zeros_like(k)
      # dv = torch.zeros_like(v)
      dq_doc = torch.empty_like(q)
      dk_doc = torch.empty_like(k)
      dv_doc = torch.empty_like(v)
      CAUSAL = True
      BLK_SLICE_FACTOR = 2
      # grid_bwd = (n_ctx//block_m, num_heads*batch, 1)
      # print(f"Grid (bwd) : {grid_bwd}")
      NUM_WARPS, NUM_STAGES = 4, 2
      BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
      RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
      arg_k = k
      arg_k = arg_k * (sm_scale * RCP_LN2)
      grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
      _attn_bwd[grid](
          q, arg_k, v, sm_scale, do, dq_doc, dk_doc, dv_doc,  #
          M, delta,  #
          q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
          N_HEAD, N_CTX,  #
          BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
          BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
          BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
          HEAD_DIM=HEAD_DIM,  #
          num_warps=NUM_WARPS,  #
          num_stages=NUM_STAGES,  #
          CAUSAL=True,  #
      )
      logging.info(f'dq : Minumum gradient : {dq_doc.dtype} : {dq_doc.min()}, Maximum gradient : {dq_doc.max()}')
      logging.info(f'dk : Minumum gradient : {dk_doc.dtype} :{dk_doc.min()}, Maximum gradient : {dk_doc.max()}')
      logging.info(f'dv : Minumum gradient : {dv_doc.dtype} :{dv_doc.min()}, Maximum gradient : {dv_doc.max()}')
      logging.info("-----------------------------")
      dq = torch.empty_like(q)
      dk = torch.empty_like(k)
      dv = torch.empty_like(v)
      BLOCK_M, BLOCK_N = 128, 32
      grid_bwd = lambda META: (
          triton.cdiv(N_CTX, BLOCK_M),
          N_HEAD * BATCH,
          1
      )
      _attention_bwd[grid_bwd](q, arg_k, v, do, dq, dk, dv, M, delta, sm_scale, BATCH, N_HEAD, N_CTX,
                               HEAD_DIM, BLOCK_M, BLOCK_N, num_warps=NUM_WARPS, num_stages=NUM_STAGES, CAUSAL=CAUSAL)
      logging.info(f'dq : Minumum gradient : {dq.dtype} : {dq.min()}, Maximum gradient : {dq.max()}')
      logging.info(f'dk : Minumum gradient : {dk.dtype} :{dk.min()}, Maximum gradient : {dk.max()}')
      logging.info(f'dv : Minumum gradient : {dv.dtype} :{dv.min()}, Maximum gradient : {dv.max()}')
      logging.info("-----------------------------")
      sys.exit()
      # if world_size != 1:
      #       exe_order_per_rank_v, exe_order_per_rank_h, exe_order_per_rank_unaligned  = identify_nodes_for_qkv(world_size)
      #       dist.barrier()
      #       for i in range(world_size):
      #         exe_order_per_rank_v[i].remove((i,i))
      #         exe_order_per_rank_h[i].remove((i,i))
      #       work_list_dq = []
      #       work_list_dk = []
      #       work_list_dv = []

      #       output_list_dq = []
      #       output_list_dk = []
      #       output_list_dv = []

      #       for _, kv_rank  in exe_order_per_rank_h[rank]:
      #         # print(f"Output({rank}) irecv src: {kv_rank}")
      #         recv_dq = torch.empty_like(dq)
      #         req_rec_dq = dist.irecv(recv_dq, src=kv_rank, tag=0)
      #         work_list_dq.append(req_rec_dq)
      #         output_list_dq.append(recv_dq)

      #         recv_dk = torch.empty_like(dk)
      #         req_rec_dk = dist.irecv(recv_dk, src=kv_rank, tag=1)
      #         work_list_dk.append(req_rec_dk)
      #         output_list_dk.append(recv_dk)

      #         recv_dv = torch.empty_like(dv)
      #         req_rec_dv = dist.irecv(recv_dv, src=kv_rank, tag=2)  
      #         work_list_dv.append(req_rec_dv)
      #         output_list_dv.append(recv_dv)

      #       for idx in range(world_size):
      #         for q_rank, kv_rank in exe_order_per_rank_unaligned[idx]:
      #           if rank == q_rank and idx != rank:
      #             # print(f"Output({rank}) irecv src: {q_rank} : {kv_rank} : {idx}")
      #             recv_dq = torch.empty_like(dq)
      #             req_rec_dq = dist.irecv(recv_dq, src=q_rank, tag=0) 
      #             work_list_dq.append(req_rec_dq)
      #             output_list_dq.append(recv_dq)

      #             recv_dk = torch.empty_like(dk)
      #             req_rec_dk = dist.irecv(recv_dk, src=kv_rank, tag=1) 
      #             work_list_dk.append(req_rec_dk)
      #             output_list_dk.append(req_rec_dk)

      #             recv_dv = torch.empty_like(dv)
      #             req_rec_dv = dist.irecv(recv_dv, src=kv_rank, tag=2)  
      #             work_list_dv.append(req_rec_dv)
      #             output_list_dv.append(recv_dv)
                        
      #       nodes_partion_q_fixed_kv_backward(exe_order_per_rank_h, exe_order_per_rank_v, rank, q, k, v, grid_preprocess, grid_bwd,
      #                                         o, do, M, pre_block, sm_scale, batch, num_heads, n_ctx, num_hiddens,
      #                                         bulk_slice_factor, warp_specialize=True)

      #       nodes_partion_vary_qkv_backward(exe_order_per_rank_unaligned, rank, q, k, v, grid_preprocess, grid_bwd,
      #                              o, do, M, dq, dk, dv, pre_block, sm_scale, batch, num_heads, n_ctx, num_hiddens, 
      #                              bulk_slice_factor, warp_specialize=True)

      #       for dq_recv, work_dq, dk_recv, work_dk, dv_recv, work_dv in zip(output_list_dq, work_list_dq, output_list_dk, work_list_dk, output_list_dv, work_list_dv):
      #         work_dq.wait()
      #         work_dk.wait()
      #         work_dv.wait()
      #         with semaphore:
      #           # print(f"Output(R:{rank}) : {dq_recv.shape}, {dk_recv.shape}, {dv_recv.shape}")
      #           dq += dq_recv
      #           dk += dk_recv
      #           dv += dv_recv
      #       dist.barrier()
      # logging.debug(f"dv : {dv.shape}")
      # logging.debug(f"dk : {dk.shape}")
      # logging.debug(f"dq : {dq.shape}")
      # logging.debug(f"[Attention] dv: {torch.isnan(dv).any()} : dv.max(): {dv.abs().max()} : dv.min(): {dv.abs().min()}")
      # logging.debug(f"[Attention] dk: {torch.isnan(dk).any()} : dk.max(): {dk.abs().max()} : dk.min(): {dk.abs().min()}")
      # logging.debug(f"[Attention] dq: {torch.isnan(dq).any()} : dq.max(): {dq.abs().max()} : dq.min(): {dq.abs().min()}")
      return dq, dk, dv, None, None, None, None, None, None, None, None, None