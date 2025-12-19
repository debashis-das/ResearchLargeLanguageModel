import torch
import torch.distributed as dist
from torch import nn
from config import Config
from transformer.RMSNorm import RMSNorm
from transformer.RopeEmbedding import RopeEmbedding
from partitioner.gpu import identify_nodes_for_qkv, nodes_partion_q_fixed_kv, nodes_partion_vary_qkv

class MultiGPUExecutor:

  def __init__(self, world_size, rank, device):
    self.device = device
    self.rank = rank
    self.world_size = world_size
    self.tokens_per_gpu = Config.tokens//world_size
    self.embedding = nn.Embedding(Config.total_vocab, Config.hiddens)
    self.rms = RMSNorm(Config.hiddens)
    self.W_q = nn.LazyLinear(Config.hiddens, bias=False)
    self.W_k = nn.LazyLinear(Config.hiddens, bias=False)
    self.W_v = nn.LazyLinear(Config.hiddens, bias=False)
    self.rope_embedding = RopeEmbedding(Config.hiddens, Config.dropout, self.tokens_per_gpu, rank)
    

  def execute(self):
    exe_order_per_rank_v, exe_order_per_rank_h, exe_order_per_rank_unaligned  = identify_nodes_for_qkv(world_size)
    sample_input = torch.randint(low=0, high=Config.total_vocab, size=(self.tokens_per_gpu,)).tolist()
    input = torch.tensor(sample_input, dtype=torch.int32)
    embedded_tensor = self.embedding(input)
    X = self.rms(embedded_tensor)
    input_q, input_k, input_v = self.W_q(X), self.W_k(X), self.W_v(X)
    input_q, input_k = self.rope_embedding(input_q, input_k)
    print(f"Input({rank}) {input_q.shape}, {input_k.shape}, {input_v.shape}")

    # input_q = torch.randn((1280, 2028), dtype=torch.float32)
    # input_k = torch.randn((1280, 2028), dtype=torch.float32)
    # input_v = torch.randn((1280, 2028), dtype=torch.float32)

    # attention(Q, K, V, num_heads, sm_scale, rank)
    exe_order_per_rank_v[rank].remove((rank,rank))
    exe_order_per_rank_h[rank].remove((rank,rank))

    dist.barrier()
    nodes_partion_q_fixed_kv(exe_order_per_rank_h, exe_order_per_rank_v,
                             self.rank, input_q, input_k, input_v)
    dist.barrier()
    nodes_partion_vary_qkv(exe_order_per_rank_unaligned, self.rank,
                           input_q, input_k, input_v)
    dist.barrier()

if __name__ == "__main__":
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  # per gpu code
  dist.init_process_group("gloo")
  world_size = dist.get_world_size()
  rank = dist.get_rank()
  multi_gpu_executor = MultiGPUExecutor(world_size, rank, device)
  multi_gpu_executor.execute()






