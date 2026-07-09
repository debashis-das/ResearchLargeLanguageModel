import torch
import torch.distributed as dist

dist.init_process_group('gloo')
outputs = torch.zeros(4, dtype=torch.float32)
world_size = dist.get_world_size()

if dist.get_rank() == 0:
  rank_object_list = torch.arange(0,4, dtype=torch.float32)
  print(f"0 : {rank_object_list}")
if dist.get_rank() == 1:
  rank_object_list = torch.arange(4,8, dtype=torch.float32)
  print(f"1 : {rank_object_list}")
if dist.get_rank() == 2:
  rank_object_list = torch.arange(8,12, dtype=torch.float32)
  print(f"2 : {rank_object_list}")
if dist.get_rank() == 3:
  rank_object_list = torch.arange(12,16, dtype=torch.float32)
  print(f"3 : {rank_object_list}")

device = torch.device("cpu")
dist.all_to_all_single(outputs, rank_object_list)
print(f"Rank : {dist.get_rank()} Output : {outputs}")
