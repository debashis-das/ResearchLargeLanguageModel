import torch
import torch.distributed as dist

dist.init_process_group('gloo')

if dist.get_rank() == 0:
  objects = ["ierjper", 232, {1: 2}]
else:
  objects = [None, None, None]

print(f"Rank : {dist.get_rank()}")
device = torch.device("cpu")
dist.broadcast_object_list(objects, src=0, device=device)
print(objects)