import torch
import torch.distributed as dist

dist.init_process_group('gloo')

if dist.get_rank() == 0:
  objects = torch.arange(0,10, dtype=torch.int32)
else:
  objects = torch.zeros((10,), dtype=torch.int32)

print(f"Rank : {dist.get_rank()}")
device = torch.device("cpu")
dist.broadcast(objects, src=0)
print(objects)