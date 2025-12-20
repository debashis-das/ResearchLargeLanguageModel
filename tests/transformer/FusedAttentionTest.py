import torch
from torch import nn

from transformer.FusedAttention import attention

def test_attention(device, q_index, kv_index):
    block_m, block_n, tokens, total_vocab, world_size, num_heads, sm_scale, embedding = configs()
    tokens_per_gpu = tokens // world_size
    sample_input = torch.randint(low=0, high=total_vocab, size=(tokens_per_gpu,)).tolist()
    input = torch.tensor(sample_input, dtype=torch.int32)
    input_q = embedding(input)
    input_k = embedding(input)
    input_v = embedding(input)
    output_o = attention(input_q, input_k, input_v, block_m, block_n, num_heads, sm_scale, device, q_index, kv_index)
    print(output_o.shape)

def configs():
    num_heads = 8
    world_size = 4
    tokens = 12*world_size
    hiddens = 8
    total_vocab = 200021
    dropout = 0.1
    sm_scale = 0.5
    block_m = 4
    block_n = 2
    embedding = nn.Embedding(total_vocab, hiddens)
    return block_m, block_n, tokens, total_vocab, world_size, num_heads, sm_scale, embedding

if __name__ == "__main__":
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  test_attention(device, 1, 0)