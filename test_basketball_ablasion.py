import NCP

### vars
SUBSPACE_DIMS = [128, 128, 128, 128]
IRRELEVANT_SUBSPACES = [3]
U_FILEPATH = '/u/kd9132/n/fs/ncp/NCP.v2/data/projection_matrices/U_basketball_tensor.pt'

### load tensor U 
U = torch.load(U_FILEPATH)
# ablate and transpose
U_ab, U_ab_T = ablate_subspace_matrix(U, SUBSPACE_DIMS, IRRELEVANT_SUBSPACES)

### augment VGG16 with ablated matrix
raise NotImplementedError

### generate LRP heatmap of augmented VGG16 matrix
