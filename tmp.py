import torch

torch.cuda.empty_cache()
print(torch.cuda.get_device_name(0)) 

