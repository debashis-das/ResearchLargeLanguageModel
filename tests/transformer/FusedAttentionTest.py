import torch
from torch import nn
# import triton

from transformer.FusedAttentionTorch import attention_bwd


# DEVICE = triton.runtime.driver.active.get_active_torch_device()

def test_attention(device, q_index, kv_index):
    block_m, block_n, tokens, hiddens, total_vocab, world_size, num_heads, sm_scale, embedding = configs()
    tokens_per_gpu = tokens // world_size
    dtype=torch.bfloat16
    q = torch.randn((num_heads, tokens_per_gpu, hiddens), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((num_heads, tokens_per_gpu, hiddens), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((num_heads, tokens_per_gpu, hiddens), dtype=dtype, device=device, requires_grad=True)
        
    # attention = _attention
    # output_o = attention.forward(q, k, v, block_m, block_n, num_heads, tokens_per_gpu, hiddens, sm_scale, DEVICE, q_index, kv_index)
    # print(output_o.shape)
    o = torch.randn((num_heads, tokens_per_gpu, hiddens), dtype=dtype, device=device, requires_grad=True)
    do = torch.randn_like(o)
    attention_bwd(q, k, v, block_m, block_n, num_heads, sm_scale, device, q_index, kv_index)

def configs():
    num_heads = 8
    world_size = 4
    tokens = 1024*world_size
    hiddens = 1024
    total_vocab = 200021
    dropout = 0.1
    sm_scale = 1.3
    block_m = 64
    block_n = 32
    embedding = nn.Embedding(total_vocab, hiddens)
    return block_m, block_n, tokens, hiddens, total_vocab, world_size, num_heads, sm_scale, embedding

if __name__ == "__main__":
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  test_attention(device, 1, 0)
  # only works on post-Ampere GPUs right now
  # bench_flash_attention.run(save_path=".", print_data=True)