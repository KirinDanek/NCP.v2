import NCP

### vars
SUBSPACE_DIMS = [128, 128, 128, 128]
IRRELEVANT_SUBSPACES = [3]
U_FILEPATH = '/u/kd9132/n/fs/ncp/NCP.v2/data/projection_matrices/U_basketball_tensor.pt'
IMAGE_FILEPATH = '/u/kd9132/n/fs/ncp/NCP.v2/data/images/drsa-basketball-img3.jpg'
CLASS_NAME = 'basketball'

### load tensor U 
U = torch.load(U_FILEPATH)
# ablate and transpose
U_ab, U_ab_T = ablate_subspace_matrix(U, SUBSPACE_DIMS, IRRELEVANT_SUBSPACES)
U_ab = U_ab.cuda()
U_ab_T = U_ab_T.cuda()

### augment VGG16 with ablated matrix
augmentedVGG16 = AugmentedVGG16(U=U_ab, UT=U_ab_T).cuda()
augmentedVGG16.eval()

### generate LRP heatmap of augmented VGG16 on drsa basketball image 3
raise NotImplementedError
