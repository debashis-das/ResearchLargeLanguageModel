from dataclasses import dataclass


@dataclass
class Config:
    debug: bool = False
    tokens = 5120
    hiddens = 2048
    total_vocab = 200021
    dropout = 0.1
