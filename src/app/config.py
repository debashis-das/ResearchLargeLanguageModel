from dataclasses import dataclass


@dataclass
class Config:
    debug: bool = False
    tokens = 64
    num_heads = 4
    # hiddens = 2048
    hiddens = 64*num_heads
    total_vocab = 200021
    dropout = 0.1
    sm_scale = 1.3
    
    #attention
    block_m = 32
    block_n = 16