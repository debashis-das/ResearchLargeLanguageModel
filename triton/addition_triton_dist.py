import torch
import torch.distributed as dist
import triton
import triton.language as tl

@triton.jit
def add_triton_inner(a_ptr, b_ptr, output_ptr, a_stride_0, a_stride_1, a_stride_2, b_stride_0, b_stride_1, b_stride_2,
                         X_DIM, Y_DIM, BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
  y_id = tl.program_id(0)
  x_id = tl.program_id(1)

  col_x = x_id*BLOCK_X
  row_y = y_id*BLOCK_Y
  offset_y = row_y + tl.arange(0, BLOCK_Y)
  offset_x = col_x + tl.arange(0, BLOCK_X)
  offset = offset_y[:, None]*a_stride_1 + offset_x[None,:]*a_stride_2
  a = tl.load(a_ptr+offset);
  b = tl.load(b_ptr+offset);
  c = a + b
  tl.store(output_ptr+offset, c)

def add_triton_dist(a, b, output, BATCH_SIZE, X_DIM, Y_DIM, BLOCK_X, BLOCK_Y):
  assert a.shape == b.shape, "Addition Impossible"
  a_stride_0 = a.stride(0)
  a_stride_1 = a.stride(1)
  a_stride_2 = a.stride(2)
  b_stride_0 = b.stride(0)
  b_stride_1 = b.stride(1)
  b_stride_2 = b.stride(2)

  y_axis = Y_DIM * BATCH_SIZE
  x_axis = X_DIM
  grid_0 = y_axis // BLOCK_Y
  grid_1 = X_DIM // BLOCK_X
  grid = (grid_0, grid_1)
  add_triton_inner[grid](a, b, output, a_stride_0, a_stride_1, a_stride_2, b_stride_0, b_stride_1, b_stride_2, X_DIM, Y_DIM, BLOCK_X, BLOCK_Y)

DEVICE = triton.runtime.driver.active.get_active_torch_device()
# DEVICE = torch.device("cuda:0")
BATCH_SIZE = 200
X_DIM = 512
Y_DIM = 512
a = torch.rand(BATCH_SIZE, X_DIM, Y_DIM, device=DEVICE, dtype=torch.float32)
b = torch.rand(BATCH_SIZE, X_DIM, Y_DIM, device=DEVICE, dtype=torch.float32)
output = torch.zeros(BATCH_SIZE, X_DIM, Y_DIM, device=DEVICE, dtype=torch.float32)
BLOCK_X = 32
BLOCK_Y = 32

# Single triton kernel
# add_triton_dist(a, b, output, BATCH_SIZE, X_DIM, Y_DIM, BLOCK_X, BLOCK_Y)
# print(output, torch.add(a,b))

def multi_triton_add(a, b, output, BATCH_SIZE, X_DIM, Y_DIM, BLOCK_X, BLOCK_Y):
  # Multi-triton kernel
  dist.init_process_group(backend='gloo')
  rank = dist.get_rank()
  world_size = dist.get_world_size()
  assert BATCH_SIZE % world_size == 0, "Batch size must be divisible by world size"
  result_list = [
    None for _ in range(world_size)
  ]

  batch_per_rank = BATCH_SIZE // world_size
  input_a = a[rank*batch_per_rank: (rank+1)*batch_per_rank,:,:]
  input_b = b[rank*batch_per_rank: (rank+1)*batch_per_rank,:,:]
  output_rank = output[rank*batch_per_rank: (rank+1)*batch_per_rank,:,:]
  add_triton_dist(input_a, input_b, output_rank, batch_per_rank, X_DIM, Y_DIM, BLOCK_X, BLOCK_Y)
  # print(output_rank.shape)
  output_rank_object = {rank : output_rank}
  dist.all_gather_object(result_list, output_rank_object)
  dist.barrier()
  if rank == 0:
    output_result = [
      torch.zeros_like(output_rank) for _ in range(world_size)
    ]
    for output_per_rank in result_list:
      for key in output_per_rank.keys():
        output_result[key] = output_per_rank[key]
    result = torch.cat(output_result)
    # print(output_rank-torch.add(input_a,input_b))
    print(result.shape)

multi_triton_add(a, b, output, BATCH_SIZE, X_DIM, Y_DIM, BLOCK_X, BLOCK_Y)




# /content# 
# /content# torchrun --standalone --nproc_per_node=4 triton_dist.py 
# W1117 16:09:06.719000 21866 torch/distributed/run.py:774] 
# W1117 16:09:06.719000 21866 torch/distributed/run.py:774] *****************************************
# W1117 16:09:06.719000 21866 torch/distributed/run.py:774] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
# W1117 16:09:06.719000 21866 torch/distributed/run.py:774] *****************************************
# [Gloo] Rank 0 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
# [Gloo] Rank 1 is connected to 3 peer ranks. Expected number of connected peer ranks is : 3
# [Gloo] Rank 3[Gloo] Rank 2 is connected to 3 peer ranks. Expected number of connected peer ranks is :  is connected to 3
# 3 peer ranks. Expected number of connected peer ranks is : 3
# torch.Size([100, 1024, 1024])
# /content#
# /content# 
