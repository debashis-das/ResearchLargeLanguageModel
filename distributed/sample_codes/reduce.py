# Common causes
# 	1.	Shared storage between processes
# 	•	You constructed the tensor before forking/spawning processes so child processes share the same underlying storage (or you explicitly used share_memory_() / torch.Tensor from a shared memory pool). A reduce that writes the dst in-place will change the shared memory and therefore appear mutated in other ranks.
# 	2.	Using threads or not using separate processes
# 	•	torch.distributed requires separate processes for each rank (or correct use of multithreading with explicit isolation). If multiple ranks run in the same Python process and reuse the same object, writes leak.
# 	3.	Indexing/aliases
# 	•	Using views/indices into a larger shared tensor so different ranks point to overlapping memory.
# 	4.	Wrong collective semantics
# 	•	reduce(..., dst=r) writes the result only on the dst (in-place). If you expected every rank to receive the sum, you should use all_reduce.
# 	5.	Backend / bug oddities (less common)
# 	•	Using an unstable backend or mixing CUDA/Gloo incorrectly. But since your Result(1) is correct this is less likely.

# Fixes — quick and reliable
# 	•	Ensure each process creates its own tensor instance (no shared storage). Easiest rules:
#   	•	Create the tensor after init_process_group() and after the process is spawned.
#   	•	Do not call .share_memory_() unless you intentionally want shared memory and you know what you’re doing.
#   	•	If you must start from a preexisting tensor, call tensor = tensor.clone() inside each process so the storage is unique.
# 	•	If you want the sum on all ranks, use torch.distributed.all_reduce(tensor, op=dist.ReduceOp.SUM) instead of reduce.
# 	•	Use torch.distributed.barrier() to synchronize before/after the collective while debugging.
# 	•	Prefer separate processes (spawn/fork) for each rank (common pattern is torch.multiprocessing.spawn).

import torch
import torch.distributed as dist

dist.init_process_group('gloo')

if dist.get_rank() == 0:
  objects = torch.arange(0,10, dtype=torch.int32) + 1 + 2 * 0
  print(f"0 : {objects}")
if dist.get_rank() == 1:
  objects = torch.arange(0,10, dtype=torch.int32) + 1 + 2 * 1
  print(f"1 : {objects}")
if dist.get_rank() == 2:
  objects = torch.arange(0,10, dtype=torch.int32) + 1 + 2 * 2
  print(f"2 : {objects}")
if dist.get_rank() == 3:
  objects = torch.arange(0,10, dtype=torch.int32) + 1 + 2 * 3
  print(f"3 : {objects}")

device = torch.device("cpu")
dist.reduce(objects,  dst=1, op=dist.ReduceOp.SUM)
print(f"Result({dist.get_rank()}) : {objects}")