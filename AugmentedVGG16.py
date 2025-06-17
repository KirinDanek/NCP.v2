import numpy as np
import torch
from torch import nn
from torchvision.models import vgg16

#### Take as input a subspace nn.Parameter U of dimension DxD, a list of 
# dimensions of each DRSA subspace, and a list of irrelevant subspace
# indices (zero indexed). Returns (new) ablated weight matrices U_ab and U_ab_transpose.
def ablate_subspace_matrix(U: torch.Tensor, subspace_dims: list, irrelevant_subspaces: list):
    U_ab = U.clone()
    U_ab_transpose = U.t().clone()

    for k_prime in irrelevant_subspaces:
        start_dim = sum(subspace_dims[:k_prime])
        block_size = subspace_dims[k_prime]
        U_ab[:, start_dim : start_dim + block_size] = 0
        U_ab_transpose[start_dim : start_dim + block_size, :] = 0

    return U_ab, U_ab_transpose


#### Given U and UT (ablated) torch.Tensors in R(512x512), insert virtual
# layer at conv4_3 (as described in Chormai et al 2024).
# ex use: 
#    U     = U_ab.cuda()
#    UT    = U_ab_transpose.cuda()
#    model = AugmentedVGG16(U=U, UT=UT).cuda()
#    model.eval()
### reference https://github.com/p16i/drsa-demo/blob/main/cxai/inspector/base.py
class AugmentedVGG16(nn.Module):
    def __init__(self, U: torch.Tensor, UT: torch.Tensor):
        """
        Augmented VGG16 inserting a virtual layer at conv4_3.
        
        Args:
            U   (torch.Tensor): Decoding matrix of shape (512, 512)
            UT  (torch.Tensor): Encoding matrix of shape (512, 512)
        """
        super().__init__()
        
        # Load pretrained VGG16 and split its feature extractor
        base = vgg16(pretrained=True)
        features = base.features  ## sequential container for conv layers
        
        # Freeze all parameters in the original feature extractor and classifier
        for param in features.parameters():
            param.requires_grad = False
        for param in base.classifier.parameters():  # classifier (fc) layers
            param.requires_grad = False
        
        # Build 1x1 convs for encoding (UT) and decoding (U)
        self.encode = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, bias=False) #1x1 conv layer --> linear layer, channelwise
        self.decode = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, bias=False)
        
        # Copy UT into encode.weight and U into decode.weight
        # UT and U should be torch.Tensors of shape (512,512)
        with torch.no_grad():
            self.encode.weight.copy_(UT.view(512, 512, 1, 1)) 
            self.decode.weight.copy_(U.view(512, 512, 1, 1))
        
        # Freeze encode/decode weights
        for param in self.encode.parameters():
            param.requires_grad = False
        for param in self.decode.parameters():
            param.requires_grad = False
        
        # Identify conv4_3 layer in VGG16 features:
        # conv4_3 is features[21] (0-indexed) followed by ReLU at [22].
        # We will apply encode/decode after the ReLU at index 22. (before maxpool)
        # So "before" includes indices 0..22, and "after" starts at 23.
        self.before = nn.Sequential(*list(features.children())[:23])  # up to ReLU after conv4_3
        self.after = nn.Sequential(*list(features.children())[23:])  # pool4 onward
        
        # Keep the original classifier unchanged
        self.classifier = base.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Pass through layers up to conv4_3's ReLU output
        x = self.before(x)  # shape: (B, 512, 28, 28) when input is 224x224
        
        # 2. Encode into subspace with UT, then decode back with U
        x = self.encode(x)  # shape becomes (B, 512, 28, 28)
        x = self.decode(x)  # returns to (B, 512, 28, 28)
        
        # 3. Continue through the remaining conv/pool layers
        x = self.after(x)
        
        # 4. Flatten and run through classifier
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
