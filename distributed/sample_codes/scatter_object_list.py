import torch
import torch.distributed as dist

dist.init_process_group('gloo')
outputs = [None]
world_size = dist.get_world_size()

if dist.get_rank() == 0:
  rank_object_list = ["foo", "psifjpsf", 12131, {1:232}]
  print(f"0 : {rank_object_list}")
if dist.get_rank() == 1:
  rank_object_list = None
  print(f"1 : {rank_object_list}")
if dist.get_rank() == 2:
  rank_object_list = None
  print(f"2 : {rank_object_list}")
if dist.get_rank() == 3:
  rank_object_list = None
  print(f"3 : {rank_object_list}")

device = torch.device("cpu")
dist.scatter_object_list(outputs, rank_object_list)
print(f"Rank : {dist.get_rank()} Output : {outputs}")