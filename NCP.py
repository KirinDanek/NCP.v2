import numpy as np
import torch
from torch import nn


#### Take as input a subspace matrix U of dimension DxD, a list of 
# dimensions of each DRSA subspace, and a list of irrelevant subspace
# indices (zero indexed). Returns ablated weight matrices U_ab and U_ab_transpose.
def ablate_subspace_matrix(U: torch.Tensor, subspace_dims: list[int], irrelevant_subspaces: list[int]):
    U_ab = U.clone()
    U_ab_transpose = U.t().clone()

    for k_prime in irrelevant_subspaces:
         start_dim = sum(subspace_dims[:k_prime])
         block_size = subspace_dims[k_prime]
         U_ab[:, start_dim : start_dim + block_size] = 0
         U_ab_transpose[start_dim : start_dim + block_size, :] = 0
         
    return U_ab, U_ab_transpose