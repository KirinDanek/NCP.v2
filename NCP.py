import numpy as np
import torch
from torch import nn


#### Take as input a subspace nn.Parameter U of dimension DxD, a list of 
# dimensions of each DRSA subspace, and a list of irrelevant subspace
# indices (zero indexed). Returns (new) ablated weight matrices U_ab and U_ab_transpose.
def ablate_subspace_matrix(U: torch.Tensor, subspace_dims: list[int], irrelevant_subspaces: list[int]):
    U_ab = U.clone()
    U_ab_transpose = U.t().clone()

    for k_prime in irrelevant_subspaces:
         start_dim = sum(subspace_dims[:k_prime])
         block_size = subspace_dims[k_prime]
         U_ab[:, start_dim : start_dim + block_size] = 0
         U_ab_transpose[start_dim : start_dim + block_size, :] = 0
         
    return U_ab, U_ab_transpose

#### Take as input DxD nn.Parameters U_ab and U_ab_transpose, pretrained 
# neural network model, and subspace extraction layer l_star. Create two
# new layers of model h1 and h2 s.t. ...->l_star->h1->h2->l_star+1->...
# where h1 = U_ab_transpose x l_star, h2 = U_ab x h1, and the original 
# weights connecting l_star to l_star+1 now connect h2 to l_star+1.
# Return modified model.
def append_layers(model: nn.Module, U_ab: torch.Tensor, U_ab_transpose:
                   torch.Tensor, l_star:str) -> nn.Module:
     


     
     return model