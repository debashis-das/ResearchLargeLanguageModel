from dataclasses import dataclass


@dataclass
class Config:
    debug: bool = False
    batch = 8
    tokens = 4096
    num_heads = 16
    # hiddens = 2048
    hiddens = 128*num_heads
    total_vocab = 200021
    dropout = 0.1
    sm_scale = 1.3
    #attention
    block_m = 128
    block_n = 128
    #mlp
    mlp_intermediate_hidden = 1024
    learning_iter = 8