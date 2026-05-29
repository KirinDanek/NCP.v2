import numpy as np
import torch

# Load the NumPy array
npy_path = "/n/fs/ncp/NCP.v2/data/projection_matrices/U_carton.npy"
array = np.load(npy_path)
print(f"[INFO] Loaded NumPy array from {npy_path} with shape: {array.shape}")
# Convert to torch tensor
tensor = torch.from_numpy(array)

tensor = tensor.reshape(512, 512)

# Save as .pt
pt_path = "/n/fs/ncp/NCP.v2/data/projection_matrices/U_carton_tensor.pt"
torch.save(tensor, pt_path)

print(f" tensor to {pt_path} with shape {tensor.shape}")


reference_pt_path = "/n/fs/ncp/NCP.v2/data/projection_matrices/U_basketball_tensor.pt"

# --- Load and compare to reference tensor ---
reference_tensor = torch.load(reference_pt_path)
print(f"[INFO] Loaded reference tensor: {reference_pt_path} | Shape: {reference_tensor.shape}")

if tensor.shape != tuple(reference_tensor.shape):
    raise ValueError(f"[ERROR] Shape mismatch! ")
else:
    print("[SUCCESS] Shape preserved during conversion.")