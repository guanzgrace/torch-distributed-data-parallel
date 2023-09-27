import numpy as np
import os
import torch
import torch.distributed as dist

# Parameters:
double_precision = False  # double precision should be slower and gains are minimal
batch_size = 50000  # adjust amount of data to fit in the GPU
epochs = 1000  # maximum amount of training epochs
learning_rate = 0.1  
# End of parameters

# Set device on GPU if available, else CPU
options = {
    "cuda": torch.has_cuda,  # cuda
    "mps": torch.has_mps,  # metal backend for mac
    "cpu": True
}
device = None
for option, check in options.items():
    if check:
        device = torch.device(option)
        print("Using device:", device)
        if option == "mps":
            double_precision = False
        break

if double_precision:
    float_np = np.float64
    float_pt = torch.float64
else:
    float_np = np.float32
    float_pt = torch.float32

# Set up and clean up the process groups
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
