import torch
import torch.nn as nn
import torchvision

print(torch.__version__)

print(torch.version.cuda)

print(torch.backends.cudnn.version())

print(torch.cuda.get_device_name(0))

# reproducibility
import numpy as np
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# GPU setting (single or multiple)
# single gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# multiple gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

# clear the cache
torch.cuda.empty_cache()

# set in the command line
# CUDA_VISIBLE_DEVICES=0,1 python train.py    --> set GPU
# nvidia-smi -- gpu-reset -i [gpu_id]    --> reset GPU

























