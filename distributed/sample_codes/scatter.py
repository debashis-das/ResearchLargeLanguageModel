import torch
import torch.distributed as dist

dist.init_process_group('gloo')
tensor_size = 5
output = torch.zeros(5, dtype=torch.int32)
world_size = dist.get_world_size()

if dist.get_rank() == 0:
  rank_object_list = [torch.arange(0,tensor_size, dtype=torch.int32) for _ in range(3)]
  print(f"0 : {rank_object_list}")
if dist.get_rank() == 1:
  rank_object_list = [torch.arange(0,tensor_size, dtype=torch.int32) for _ in range(1)]
  print(f"1 : {rank_object_list}")
if dist.get_rank() == 2:
  rank_object_list = None
  print(f"2 : {rank_object_list}")
if dist.get_rank() == 3:
  rank_object_list = None
  print(f"3 : {rank_object_list}")

device = torch.device("cpu")
dist.scatter(output, rank_object_list)
print(f"Rank : {dist.get_rank()} Output : {output}")


# 2 : tensor([[-0.6093, -0.9827,  0.7202,  0.6940,  0.1268],
#         [ 0.0742, -0.5802, -0.1072, -0.0157, -0.2996],
#         [-0.4235,  0.2548,  0.1186, -1.9886,  1.3553],
#         [ 0.2203, -0.2445,  1.0694,  0.4104,  2.1866]])0 : tensor([[-0.8076,  1.6471, -0.8607, -0.0326, -0.2798],
#         [-1.1202, -1.0312,  0.1040,  0.0115, -1.8692],
#         [ 0.4763, -0.5499, -1.4825,  0.2557, -1.0419],
#         [ 1.3148,  0.7220, -2.0368,  1.4426,  1.2100]])

# 3 : tensor([[-0.6202,  0.8835,  0.4403,  0.4232, -1.4332],
#         [-0.2980,  1.4811, -0.6230, -0.8033, -0.8080],
#         [ 0.4588,  1.2202, -0.7849,  0.0974, -1.9366],
#         [-0.4724,  0.2913,  1.0859, -1.7408,  0.9276]])
# 1 : tensor([[-0.8052, -2.0358,  0.2154,  0.6824,  1.4747],
#         [ 0.1566,  0.9333, -0.1992, -0.9145, -0.0972],
#         [ 0.3839, -0.4704, -3.0563,  0.6886,  0.7224],
#         [-0.5127,  0.9840, -0.2947, -0.2842,  1.1708]])
# Rank : 3 Output : tensor([[ 0.5500,  1.7528, -0.1762, -0.1720,  5.4950]])Rank : 0 Output : tensor([[-2.8423, -0.4879,  0.5152,  1.7670, -0.1114]])

# Rank : 2 Output : tensor([[ 0.8954,  0.4547, -5.2051, -0.9469, -0.9008]])Rank : 1 Output : tensor([[-1.1873,  0.8030, -0.8254, -1.7220, -3.0740]])

