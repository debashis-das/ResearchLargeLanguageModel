from dataclasses import dataclass


@dataclass
class Config:
    debug: bool = False
    tokens = 512
    # hiddens = 2048
    hiddens = 128
    total_vocab = 200021
    dropout = 0.1
    num_heads = 8
    sm_scale = 1.3
    
    #attention
    num_heads = 8
    block_m = 64
    block_n = 32
