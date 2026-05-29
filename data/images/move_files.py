import os
import shutil
from pathlib import Path
#import sys
from PIL import Image
#sys.path.append('/n/fs/ncp/watermark_imagenet')
#from watermark_transform import AddWatermark
#import torchvision.transforms as transforms

# --- Configuration ---
src_dir = Path("/n/fs/ncp/NCP.v2/data/images/carton_dugong/original/n02074367")
dst_dir = Path("/n/fs/ncp/NCP.v2/data/images/carton_dugong/test_set/dugong_watermarked")
dst_dir.mkdir(parents=True, exist_ok=True)
'''
# Watermark parameters
resize_size = 256
crop_size = 224 ## resize and crop done in dataloader. If synthetically
# adding watermarks, resize and crop non-watermarked images here, and
# remove resize and crop from dataloader.

transform_noop = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.CenterCrop(crop_size)
])

# --- Process ---
subdirs = sorted([d for d in src_dir.iterdir() if d.is_dir()])
non_target_imgs = []
for subdir in subdirs:
    non_target_imgs.extend(sorted(subdir.glob("*.JPEG"))[:500])  # specify indices
'''
'''
subdir = src_dir
non_target_imgs = sorted(subdir.glob("*.JPEG"))[:500]
'''
# remember to specify indices
imgs = sorted(f for f in os.listdir(src_dir) if f.endswith(".jpeg"))[500:1000]

for f in imgs:
    shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
'''for i, img_path in enumerate(imgs):
    img = Image.open(img_path).convert("RGB")

    #img = transform_noop(img)
    save_path = dst_dir / img_path.name
    img.save(save_path)
    print(f"Saved {save_path}")
'''