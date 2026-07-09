import torch
import torch.distributed as dist
from torch import nn

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

class RMSNorm(nn.Module):
    def __init__(self, num_hiddens, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_hiddens))
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
        theta = torch.pow(self.n, -2*torch.arange(1,num_hiddens//2+1,dtype=torch.float64)/num_hiddens)
        expression = torch.arange(block_size_per_gpu*rank+1, block_size_per_gpu*(rank+1)+1, dtype=torch.float64).reshape(-1,1)*theta
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


def execute():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  # per gpu code
  dist.init_process_group("gloo")
  world_size = dist.get_world_size()
  exe_order_per_rank_v, exe_order_per_rank_h, exe_order_per_rank_unaligned  = identify_nodes_for_qkv(world_size)
  rank = dist.get_rank()

  tokens = 5120
  hiddens = 2048
  total_vocab = 200021
  tokens_per_gpu = tokens//world_size
  dropout = 0.1
  embedding = nn.Embedding(total_vocab, hiddens)
  rms = RMSNorm(hiddens)
  W_q = nn.LazyLinear(hiddens, bias=False)
  W_k = nn.LazyLinear(hiddens, bias=False)
  W_v = nn.LazyLinear(hiddens, bias=False)
  rope_embedding = RopeEmbedding(hiddens, dropout, tokens_per_gpu, rank)
  sample_input = torch.randint(low=0, high=total_vocab, size=(tokens_per_gpu,)).tolist()
  input = torch.tensor(sample_input, dtype=torch.int32)
  embedded_tensor = embedding(input)
  X = rms(embedded_tensor)
  input_q, input_k, input_v = W_q(X), W_k(X), W_v(X)
  input_q, input_k = rope_embedding(input_q, input_k)
  print(f"Input({rank}) {input_q.shape}, {input_k.shape}, {input_v.shape}")

  # input_q = torch.randn((1280, 2028), dtype=torch.float32)
  # input_k = torch.randn((1280, 2028), dtype=torch.float32)
  # input_v = torch.randn((1280, 2028), dtype=torch.float32)

  # attention(Q, K, V, num_heads, sm_scale, rank)
  exe_order_per_rank_v[rank].remove((rank,rank))
  exe_order_per_rank_h[rank].remove((rank,rank))

  dist.barrier()

  recv_q = torch.empty_like(input_q)
  for (q_rank, dst_rank) in exe_order_per_rank_h[rank]:
      if q_rank != dst_rank:
          req_rec_q = dist.isend(input_q, dst=dst_rank)
          req_rec_q.wait()
          
  for (q_rank, kv_rank) in exe_order_per_rank_v[rank]:
      if kv_rank == rank:
          req_rec_q = dist.irecv(recv_q, src=q_rank)
          req_rec_q.wait()
          print(f"Rec1(s:{q_rank},c:{rank}) {recv_q.shape}, {input_k.shape}, {input_v.shape}")

  dist.barrier()

  recv_k = torch.empty_like(input_k)
  recv_v = torch.empty_like(input_v)

  input_2_send = False
  input_2_recv = False
  for rank_in_list, list_per_rank in enumerate(exe_order_per_rank_unaligned):
    for (q_rank, kv_rank) in list_per_rank:
      if q_rank == rank and rank_in_list != rank:
        req_rec_q = dist.isend(input_q, dst=rank_in_list)
        req_rec_q.wait()
        # print(f"input send (send from:{rank},dst:{rank_in_list})")
      if kv_rank == rank and rank_in_list != rank and not input_2_send:
        req_rec_k = dist.isend(input_k, dst=rank_in_list)
        req_rec_v = dist.isend(input_v, dst=rank_in_list)
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
        recv_q = torch.empty_like(input_q)
        if q_rank == rank:
          recv_q = input_q
        else:
          req_rec_q = dist.irecv(recv_q, src=q_rank)
          req_rec_q.wait()
          # print(f"input irecv (src:{q_rank},dest recevied to :{rank})")
        print(f"R({rank})(s1:{q_rank},s2:{kv_rank}) {recv_q.shape}, {recv_k.shape}, {recv_v.shape}")
  dist.barrier()

if __name__ == "__main__":
  execute()






