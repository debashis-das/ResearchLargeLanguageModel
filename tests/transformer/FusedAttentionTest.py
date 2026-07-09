import torch
from torch import nn
# import triton

from transformer.FusedAttentionTorch import _attention_forward
# from transformer.FusedAttention import _attention

# DEVICE = triton.runtime.driver.active.get_active_torch_device()

def test_attention(device, q_index, kv_index):
    attention_fn = _attention.apply
    block_m, block_n, tokens, hiddens, total_vocab, world_size, num_heads, sm_scale, embedding = configs()
    tokens_per_gpu = tokens // world_size
    dtype=torch.bfloat16
    q = torch.randn((tokens_per_gpu, hiddens), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((tokens_per_gpu, hiddens), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((tokens_per_gpu, hiddens), dtype=dtype, device=device, requires_grad=True)
    o = torch.rand_like(q)
    M = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)  
    output_o = attention_fn(q, k, v, block_m, block_n, num_heads, tokens_per_gpu,
                                        hiddens, sm_scale, device, q_index, kv_index)
    print(output_o.shape)

def test_attention_torch(device, q_index, kv_index):
    block_m, block_n, tokens, hiddens, total_vocab, world_size, num_heads, sm_scale, embedding = configs()
    tokens_per_gpu = tokens // world_size
    dtype=torch.bfloat16
    q = torch.randn((num_heads, tokens_per_gpu, hiddens), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((num_heads, tokens_per_gpu, hiddens), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((num_heads, tokens_per_gpu, hiddens), dtype=dtype, device=device, requires_grad=True)
    o = torch.rand_like(q)
    M = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)  
    print(f"q:{q.shape}, k:{k.shape}, v:{v.shape}, tokens_per_gpu: {tokens_per_gpu}")
    _attention_forward(sm_scale, M, num_heads, tokens_per_gpu, q, k, v, o, hiddens, block_m, block_n, True, True)
    print(o.shape)
    
def configs():
    num_heads = 2
    world_size = 4
    tokens = 1024*world_size
    hiddens = 128
    total_vocab = 200021
    dropout = 0.1
    sm_scale = 1.3
    block_m = 64
    block_n = 32
    embedding = nn.Embedding(total_vocab, hiddens)
    return block_m, block_n, tokens, hiddens, total_vocab, world_size, num_heads, sm_scale, embedding

if __name__ == "__main__":
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  test_attention_torch(device, 0, 0)
  # only works on post-Ampere GPUs right now
  # bench_flash_attention.run(save_path=".", print_data=True)