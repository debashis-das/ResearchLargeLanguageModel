# Overflow happens when the values are not casted properly and go negative
#
# 1 : tensor([90000002, 90000003, 90000004, 90000005, 90000006, 90000007, 90000008,
#         90000009, 90000010, 90000011], dtype=torch.int32)[Gloo] Rank 2 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
# Rank : 1
# 0 : tensor([90000000, 90000001, 90000002, 90000003, 90000004, 90000005, 90000006,
#         90000007, 90000008, 90000009], dtype=torch.int32)
# Rank : 0
# 3 : tensor([90000006, 90000007, 90000008, 90000009, 90000010, 90000011, 90000012,
#         90000013, 90000014, 90000015], dtype=torch.int32)
# Rank : 3
# 2 : tensor([90000004, 90000005, 90000006, 90000007, 90000008, 90000009, 90000010,
#         90000011, 90000012, 90000013], dtype=torch.int32)
# Rank : 2
# Result : tensor([  -84871168,  2024585321, -1987090048, -1369962575,  1741000576,
#           915864969, -1685369216,   392265521,   718834816,  1454338921],
#        dtype=torch.int32)
# Result : tensor([  -84871168,  2024585321, -1987090048, -1369962575,  1741000576,
#           915864969, -1685369216,   392265521,   718834816,  1454338921],
#        dtype=torch.int32)
# Result : tensor([  -84871168,  2024585321, -1987090048, -1369962575,  1741000576,
#           915864969, -1685369216,   392265521,   718834816,  1454338921],
#        dtype=torch.int32)
# Result : tensor([  -84871168,  2024585321, -1987090048, -1369962575,  1741000576,
#           915864969, -1685369216,   392265521,   718834816,  1454338921],
#        dtype=torch.int32)

# Use log sum exp trick for larger number multiplcation

import torch
import torch.distributed as dist

dist.init_process_group('gloo')

if dist.get_rank() == 0:
  objects = torch.arange(0,10, dtype=torch.int32) + 900 + 2 * 0
  print(f"0 : {objects}")
if dist.get_rank() == 1:
  objects = torch.arange(0,10, dtype=torch.int32) + 900 + 2 * 1
  print(f"1 : {objects}")
if dist.get_rank() == 2:
  objects = torch.arange(0,10, dtype=torch.int32) + 900 + 2 * 2
  print(f"2 : {objects}")
if dist.get_rank() == 3:
  objects = torch.arange(0,10, dtype=torch.int32) + 900 + 2 * 3
  print(f"3 : {objects}")
print(f"Rank : {dist.get_rank()}")
device = torch.device("cpu")
logObject = torch.log(objects)
dist.all_reduce(logObject, op=dist.ReduceOp.SUM)
result = torch.exp(logObject)
result_64 = torch.round(result).to(torch.int64)
print(f"Result : {result_64}")