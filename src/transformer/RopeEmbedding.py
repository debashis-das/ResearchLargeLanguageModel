import torch
from torch import nn

class RopeEmbedding(nn.Module):
    def __init__(self, num_hiddens, dropout, block_size_per_gpu, rank, device):
        super().__init__()
        # n = 10000 as per paper
        self.n = 10000
        self.dropout = nn.Dropout(dropout)
        # k/n^(2i/d) with n = 10000
        theta = torch.pow(self.n, -2*torch.arange(0,num_hiddens//2, device=device)/num_hiddens)
        expression = torch.arange(block_size_per_gpu*rank, block_size_per_gpu*(rank+1), device=device).reshape(-1,1)*theta
        # print(f'Theta : {theta}')
        sin_val = torch.sin(expression)
        cos_val = torch.cos(expression)
        # print(f'sin_val : {sin_val}')
        # print(f'cos_val : {cos_val}')
        # self.sin = sin_val.repeat_interleave(2, dim=-1)
        # self.cos = cos_val.repeat_interleave(2, dim=-1)
        self.register_buffer("sin", sin_val.repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0))
        self.register_buffer("cos", cos_val.repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0))
        # self.sin = self.sin.unsqueeze(0)
        # self.cos = self.cos.unsqueeze(0)

    # @torch.compile
    def forward(self,query,key):
        def rotate_half(x):
            x1 = x[...,::2]
            x2 = x[...,1::2]
            # result = torch.cat((-x2,x1), dim=-1).squeeze()
            return torch.stack((-x2, x1), dim=-1).flatten(-2).contiguous()
            # print(f'x1: {x1.shape}, x2: {x2.shape}, result: {result.shape}')
            # return result
        # print(f'Query({rank}) : {query.shape} , Cos : {self.cos.shape}, Rotate half : {rotate_half(query).shape}, Sin : {self.sin.shape}')
        q_type = query.dtype
        k_type = key.dtype
        q_embed = (query.float()*self.cos) + (rotate_half(query).float()*self.sin)
        k_embed = (key.float()*self.cos) + (rotate_half(key).float()*self.sin)
        return q_embed.to(q_type), k_embed.to(k_type)