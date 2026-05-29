import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
from watermark_transform import AddWatermark

# Constants
IMAGE_PATH = '/u/kd9132/n/fs/ncp/NCP.v2/data/images/imagenet_430_binary/not_basketball/n01498041/n01498041_1.JPEG'
OUTPUT_PATH = './watermarked_image.jpg'  # Change as desired
resize_size = 256
crop_size = 224

# Normalize using standard ImageNet mean/std
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Watermark transformation
watermark_transform = AddWatermark(
    image_size=crop_size,
    text="捷径捷径捷径"
)

# Compose full transformation
transform = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    watermark_transform,
    normalize
])

# Load and transform image
image = Image.open(IMAGE_PATH).convert('RGB')
transformed_tensor = transform(image)

# To visualize (unnormalize first)
unnormalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(normalize.mean, normalize.std)],
    std=[1/s for s in normalize.std]
)
visual_tensor = unnormalize(transformed_tensor).clamp(0, 1)
visual_img = to_pil_image(visual_tensor)
visual_img.save(OUTPUT_PATH)

print(f"Watermarked image saved to {OUTPUT_PATH}")
