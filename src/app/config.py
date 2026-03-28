from dataclasses import dataclass


@dataclass
class Config:
    debug: bool = False
    batch = 1
    # tokens = 4096
    tokens = 512
    # num_heads = 16
    num_heads = 4
    # hiddens = 2048
    # hiddens = 128*num_heads
    hiddens = 64*num_heads
    total_vocab = 200021
    dropout = 0.1
    sm_scale = 1.3
    #mlp
    mlp_intermediate_hidden = 512
    learning_iter = 8