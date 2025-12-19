import torch
import torch.distributed as dist

dist.init_process_group('gloo')
outputs = torch.zeros(5, dtype=torch.int32)
world_size = dist.get_world_size()

if dist.get_rank() == 0:
  rank_object_list = [torch.arange(0,5, dtype=torch.int32)+i+2*dist.get_rank() for i in range(world_size)]
  print(f"0 : {rank_object_list}")
if dist.get_rank() == 1:
  rank_object_list = [torch.arange(0,5, dtype=torch.int32)+i+2*dist.get_rank() for i in range(world_size)]
  print(f"1 : {rank_object_list}")
if dist.get_rank() == 2:
  rank_object_list = [torch.arange(0,5, dtype=torch.int32)+i+2*dist.get_rank() for i in range(world_size)]
  print(f"2 : {rank_object_list}")
if dist.get_rank() == 3:
  rank_object_list = [torch.arange(0,5, dtype=torch.int32)+i+2*dist.get_rank() for i in range(world_size)]
  print(f"3 : {rank_object_list}")

device = torch.device("cpu")
dist.reduce_scatter(outputs, rank_object_list, op=dist.ReduceOp.SUM)
print(f"Rank : {dist.get_rank()} Output : {outputs}")

# 2 : [tensor([4, 5, 6, 7, 8], dtype=torch.int32),
#     tensor([5, 6, 7, 8, 9], dtype=torch.int32),
#     tensor([ 6,  7,  8,  9, 10], dtype=torch.int32),
#       tensor([ 7,  8,  9, 10, 11], dtype=torch.int32)]

# 3 : [tensor([ 6,  7,  8,  9, 10], dtype=torch.int32),
#       tensor([ 7,  8,  9, 10, 11], dtype=torch.int32),
#         tensor([ 8,  9, 10, 11, 12], dtype=torch.int32),
#           tensor([ 9, 10, 11, 12, 13], dtype=torch.int32)]

# 0 : [tensor([0, 1, 2, 3, 4], dtype=torch.int32),
#       tensor([1, 2, 3, 4, 5], dtype=torch.int32), 
#       tensor([2, 3, 4, 5, 6], dtype=torch.int32),
#         tensor([3, 4, 5, 6, 7], dtype=torch.int32)]

# 1 : [tensor([2, 3, 4, 5, 6], dtype=torch.int32),
#       tensor([3, 4, 5, 6, 7], dtype=torch.int32),
#         tensor([4, 5, 6, 7, 8], dtype=torch.int32),
#           tensor([5, 6, 7, 8, 9], dtype=torch.int32)]

# Rank : 0 Output : tensor([12, 16, 20, 24, 28], dtype=torch.int32)
# Rank : 3 Output : tensor([24, 28, 32, 36, 40], dtype=torch.int32)
# Rank : 2 Output : tensor([20, 24, 28, 32, 36], dtype=torch.int32)
# Rank : 1 Output : tensor([16, 20, 24, 28, 32], dtype=torch.int32)
