import torch
import torch.distributed as dist

dist.init_process_group('gloo')
outputs = torch.zeros((1,5), dtype=torch.float32)
world_size = dist.get_world_size()

if dist.get_rank() == 0:
  rank_object_list = torch.randn(world_size,5, dtype=torch.float32)
  print(f"0 : {rank_object_list}")
if dist.get_rank() == 1:
  rank_object_list = torch.randn(world_size,5, dtype=torch.float32)
  print(f"1 : {rank_object_list}")
if dist.get_rank() == 2:
  rank_object_list = torch.randn(world_size,5, dtype=torch.float32)
  print(f"2 : {rank_object_list}")
if dist.get_rank() == 3:
  rank_object_list = torch.randn(world_size,5, dtype=torch.float32)
  print(f"3 : {rank_object_list}")

device = torch.device("cpu")
dist.reduce_scatter_tensor(outputs, rank_object_list, op=dist.ReduceOp.SUM)
print(f"Rank : {dist.get_rank()} Output : {outputs}")