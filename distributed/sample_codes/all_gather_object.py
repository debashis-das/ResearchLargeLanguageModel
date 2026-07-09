import torch
import torch.distributed as dist

dist.init_process_group('gloo')
output = [None for _ in range(torch.distributed.get_world_size())]

if dist.get_rank() == 0:
  rankobject = "foo0"
  print(f"0 : {rankobject}")
if dist.get_rank() == 1:
  rankobject = 123
  print(f"1 : {rankobject}")
if dist.get_rank() == 2:
  rankobject = {1: 123144}
  print(f"2 : {rankobject}")
if dist.get_rank() == 3:
  rankobject = "dfssdf"
  print(f"3 : {rankobject}")
print(f"Rank : {dist.get_rank()}")

device = torch.device("cpu")
dist.all_gather_object(output, rankobject)
print(f"Output : {output}")