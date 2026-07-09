import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoTokenizer

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
    for start_m in range(20):
        for off_h in range(1):
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
            # if mask_region:
            _attention_forward_inner_mask(acc, l_i, m_i, desc_q, desc_k, desc_v, offset_y, dtype, start_m, qk_scale, block_m, hidden_dim, block_n, 1, offs_m, offs_n, n_ctx, wrap_specialize)
            _attention_forward_inner_mask(acc, l_i, m_i, desc_q, desc_k, desc_v, offset_y, dtype, start_m, qk_scale, block_m, hidden_dim, block_n, 2, offs_m, offs_n, n_ctx, wrap_specialize)
            # else:
            #     _attention_forward_inner_non_mask(acc, l_i, m_i, q, desc_k, desc_v, offset_y, dtype, start_m, qk_scale, block_m, hidden_dim, block_n, offs_m, offs_n, n_ctx, wrap_specialize)
             
            if off_h == 1:
                break
        if start_m == 1:
            print("------------------------------")
            return

# Assumption that it is used only for causal case
def attention(q, k, v, num_heads, sm_scale, rank, wrap_specialize=True):
    HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
    q_with_head = q.unsqueeze(0).expand(num_heads, -1, -1)
    k_with_head = k.unsqueeze(0).expand(num_heads, -1, -1)
    v_with_head = v.unsqueeze(0).expand(num_heads, -1, -1)
    o_with_head = torch.empty_like(q_with_head)
    print(q_with_head.shape, k_with_head.shape, v_with_head.shape)
    BLOCK_M = 64
    BLOCK_N = 32
    M = torch.empty(q_with_head.shape[0], q_with_head.shape[1], device=device, dtype=torch.float32)
    print(f"Max tokens : {M.shape}")
    grid = (q_with_head.shape[1]//BLOCK_M, num_heads, 1)
    print(f"Grid : {grid}")
    # Attention forward
        # mask region
    _attention_forward(sm_scale, M, num_heads, q_with_head.shape[1],
                    q_with_head, k_with_head, v_with_head, o_with_head,
                    q_with_head.shape[-1], BLOCK_M, BLOCK_N, True, wrap_specialize)
    # non-mask region
    _attention_forward(sm_scale, M, num_heads, q_with_head.shape[1],
                    q_with_head, k_with_head, v_with_head, o_with_head,
                    q_with_head.shape[-1], BLOCK_M, BLOCK_N, False, wrap_specialize)

class RMSNorm(nn.Module):
    def __init__(self, num_hiddens, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_hiddens,device=device))
        self.variance_epsilon = eps
    
    # @torch.compile
    def forward(self, hidden_state: torch.Tensor):
        input_dtype = hidden_state.dtype
        hidden_state = hidden_state.to(torch.float32)
        variance = hidden_state.pow(2).mean(-1, keepdim=True)
        hidden_state = hidden_state*torch.rsqrt(variance+self.variance_epsilon)
        return self.weight*hidden_state.to(input_dtype)

class RopeEmbedding(nn.Module):
    def __init__(self, num_hiddens, dropout, block_size_per_gpu, rank):
        super().__init__()
        # n = 10000 as per paper
        self.n = 10000
        self.dropout = nn.Dropout(dropout)
        # k/n^(2i/d) with n = 10000
        theta = torch.pow(self.n, -2*torch.arange(1,num_hiddens//2+1,dtype=torch.float64,device=device)/num_hiddens)
        expression = torch.arange(block_size_per_gpu*rank+1, block_size_per_gpu*(rank+1)+1, dtype=torch.float64,device=device).reshape(-1,1)*theta
        # print(f'Theta : {theta}')
        sin_val = torch.sin(expression)
        cos_val = torch.cos(expression)
        # print(f'sin_val : {sin_val}')
        # print(f'cos_val : {cos_val}')

        self.sin = sin_val.repeat_interleave(2, dim=-1)
        self.cos = cos_val.repeat_interleave(2, dim=-1)
        # self.sin = self.sin.unsqueeze(0)
        # self.cos = self.cos.unsqueeze(0)

    # @torch.compile
    def forward(self,query,key):
        def rotate_half(x):
            x1 = x[...,:x.shape[-1]:2].unsqueeze(1)
            x2 = x[...,1:x.shape[-1]:2].unsqueeze(1)
            result = torch.cat((-x2,x1), dim=-1).squeeze()
            # print(f'x1: {x1.shape}, x2: {x2.shape}, result: {result.shape}')
            return result
        # print(f'Query({rank}) : {query.shape} , Cos : {self.cos.shape}, Rotate half : {rotate_half(query).shape}, Sin : {self.sin.shape}')
        q_type = query.dtype
        k_type = key.dtype
        q_embed = (query.float()*self.cos.float()) + (rotate_half(query).float()*self.sin.float())
        k_embed = (key.float()*self.cos.float()) + (rotate_half(key).float()*self.sin.float())
        return q_embed.to(q_type), k_embed.to(k_type)

num_heads = 8
sm_scale = 0.5

tokens = 5120
hiddens = 2048
total_vocab = 200021
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding = nn.Embedding(total_vocab, hiddens, device=device)
rms = RMSNorm(hiddens)
W_q = nn.LazyLinear(hiddens, bias=False, device=device)
W_k = nn.LazyLinear(hiddens, bias=False, device=device)
W_v = nn.LazyLinear(hiddens, bias=False, device=device)
# per gpu code
dist.init_process_group("gloo")
world_size = dist.get_world_size()
exe_order_per_rank_v, exe_order_per_rank_h, exe_order_per_rank_unaligned  = identify_nodes_for_qkv(world_size)
tokens_per_gpu = tokens//world_size
output = [None]
rank = dist.get_rank()
rope_embedding = RopeEmbedding(hiddens, dropout, tokens_per_gpu, rank)
# print(tokens_per_gpu)
if rank == 0:
    sample_input = torch.randint(low=0, high=total_vocab, size=(tokens,), device=device).tolist()
    sample_per_gpu = [{idx:sample_input[i:i+tokens_per_gpu]} for idx,i in enumerate(range(0, tokens, tokens_per_gpu))]
else:
    sample_per_gpu = None
dist.scatter_object_list(output, sample_per_gpu)
dist.barrier()
# if rank == 0:
input_tensor = torch.tensor(output[0][rank], dtype=torch.int32, device=device)
print(f"Rank {rank} input_tensor : {input_tensor.shape}")
embedded_tensor = embedding(input_tensor)
print(f"Rank {rank} embedded_tensor : {embedded_tensor.shape}")
# rms
X = rms(embedded_tensor)
print(f"Rank {rank} X : {X.shape}")
# linear
Q, K, V = W_q(X), W_k(X), W_v(X)
print(f"Rank {rank} Q, K, V : {Q.shape}, {K.shape}, {V.shape}")
# rope_embedding
Q, K = rope_embedding(Q, K)
print(f"Rank {rank} Q, K : {Q.shape}, {K.shape}")

attention(Q, K, V, num_heads, sm_scale, rank)
exe_order_per_rank_v[rank].remove((rank,rank))
exe_order_per_rank_h[rank].remove((rank,rank))

dist.barrier()

recv_q = torch.empty_like(Q)
req_rec = None

for (q_rank, dst_rank) in exe_order_per_rank_h[rank]:
    if q_rank != dst_rank:
        req_rec = dist.isend(Q, dst=dst_rank)
        req_rec.wait()
for (q_rank, kv_rank) in exe_order_per_rank_v[rank]:
    if kv_rank == rank:
        req_rec = dist.irecv(recv_q, src=q_rank)
        req_rec.wait()
        attention(recv_q, K, V, num_heads, sm_scale, rank)
        # print(f"Rec1(s:{q_rank},c:{rank}) {recv_q.shape}, {K.shape}, {V.shape}")

dist.barrier()

recv_k = torch.empty_like(K)
recv_v = torch.empty_like(V)
input_kv_send = False
input_kv_recv = False
req_rec_q = None
req_rec_k = None
req_rec_v = None

for rank_in_list, list_per_rank in enumerate(exe_order_per_rank_unaligned):
  for (q_rank, kv_rank) in list_per_rank:
    if q_rank == rank and rank_in_list != rank:
      req_rec_q = dist.isend(Q, dst=rank_in_list)
      req_rec_q.wait()
      # print(f"input send (send from:{rank},dst:{rank_in_list})")
    if kv_rank == rank and rank_in_list != rank and not input_kv_send:
      req_rec_k = dist.isend(K, dst=rank_in_list)
      req_rec_v = dist.isend(V, dst=rank_in_list)
      req_rec_k.wait()
      req_rec_v.wait()
      # print(f"input2 send (send from:{rank},dst:{rank_in_list})")
      input_kv_send = True
  
  if len(list_per_rank) != 0 and rank_in_list == rank :
    if not input_kv_recv:
      req_rec_k = dist.irecv(recv_k, src=list_per_rank[0][1])
      req_rec_v = dist.irecv(recv_v, src=list_per_rank[0][1])
      req_rec_k.wait()
      req_rec_v.wait()
      # print(f"input2 irecv (src:{list_per_rank[0][1]},dest recevied to :{rank})")
      input_kv_recv = True
    for (q_rank, kv_rank) in list_per_rank:
      recv_q = torch.empty_like(Q)
      if q_rank == rank:
        recv_q = Q
      else:
        req_rec_q = dist.irecv(recv_q, src=q_rank)
        req_rec_q.wait()
        # print(f"input irecv (src:{q_rank},dest recevied to :{rank})")
      attention(recv_q, K, V, num_heads, sm_scale, rank)
    #   print(f"R({rank})(s1:{q_rank},s2:{kv_rank}) {recv_q.shape}, {recv_k.shape}, {recv_v.shape}")
dist.barrier()








