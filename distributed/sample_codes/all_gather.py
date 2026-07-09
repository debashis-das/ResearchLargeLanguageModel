import torch
import torch.distributed as dist

dist.init_process_group('gloo')
device = torch.device("cpu")

if dist.get_rank() == 0:
  tensor = torch.arange(0,10, dtype=torch.int32) + 0 + 2 * 0
  print(f"0 : {tensor}")
if dist.get_rank() == 1:
  tensor = torch.arange(0,10, dtype=torch.int32) + 1 + 2 * 0
  print(f"1 : {tensor}")
if dist.get_rank() == 2:
  tensor = torch.arange(0,10, dtype=torch.int32) + 2 + 2 * 0
  print(f"2 : {tensor}")
if dist.get_rank() == 3:
  tensor = torch.arange(0,10, dtype=torch.int32) + 3 + 2 * 0
  print(f"3 : {tensor}")

result_list = [  
  torch.zeros(10, dtype=torch.int32) for _ in range(4)  
]
dist.all_gather(result_list, tensor)
print(f"Result({dist.get_rank()}) : {result_list}")