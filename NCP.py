import numpy as np


#### Take as input a subspace matrix U of dimension DxD, a list of 
# dimensions of each DRSA subspace, and a list of irrelevant subspace
# indices. Returns ablated weight matrices U_ab and U_ab_transpose.
def ablate_subspace_matrix(U, subspace_dims, irrelevant_subspaces):
    U_ab = U.copy()
    U_ab_transpose = U.copy().T

    for k_prime in irrelevant_subspaces:
         startdim=0
         for i in range(k_prime - 1): # -1 since matrices are 0-indexed
              startdim += subspace_dims[i]
         U_ab[:, startdim : startdim + subspace_dims[k_prime]] = 0
         U_ab_transpose[startdim : startdim + subspace_dims[k_prime]] = 0
         
    return U_ab, U_ab_transpose