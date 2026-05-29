import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from watermark_transform import AddWatermark
from pathlib import Path

# --- Config ---
INPUT_DIR = Path("/u/kd9132/n/fs/ncp/NCP.v2/data/images/imagenet_n03127925_binary/target")
OUTPUT_DIR = INPUT_DIR.parent / "target_watermarked"
NUM_IMAGES = 125
RESIZE_SIZE = 256
CROP_SIZE = 224

# --- Transforms ---
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

watermark_transform = AddWatermark(
    image_size=CROP_SIZE,
    text="捷径捷径捷径"
)

transform = transforms.Compose([
    transforms.Resize(RESIZE_SIZE),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    watermark_transform,
    normalize
])

unnormalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(normalize.mean, normalize.std)],
    std=[1 / s for s in normalize.std]
)

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
image_files = sorted([f for f in INPUT_DIR.iterdir() if f.suffix.lower() == ".jpeg"])[:NUM_IMAGES]

# --- Main loop ---
for i, image_path in enumerate(image_files):
    try:
        image = Image.open(image_path).convert("RGB")
        transformed_tensor = transform(image)
        visual_tensor = unnormalize(transformed_tensor).clamp(0, 1)
        visual_img = to_pil_image(visual_tensor)

        output_path = OUTPUT_DIR / image_path.name
        visual_img.save(output_path)
        print(f"[{i+1}/{NUM_IMAGES}] Saved: {output_path}")

    except Exception as e:
        print(f"[{i+1}] Failed on {image_path}: {e}")
