import os
import shutil
from pathlib import Path
import sys
from PIL import Image
import random
sys.path.append('/n/fs/ncp/watermark_imagenet')
from watermark_transform import AddWatermark
import torchvision.transforms as transforms

# --- Configuration ---
# NOTE : random samples target but not non target
binary_class = 'carton' #or non-target
wm_percent = 0.0 #0.0, 0.25, 0.50, 0.75, 1.0
#src_dir = Path(f"/n/fs/ncp/NCP.v2/data/images/imagenet_n03127925_binary/{binary_class}")
dst_dir = Path(f"/n/fs/ncp/NCP.v2/data/images/carton_dugong/test_set/{binary_class}") #25p means 25% of not-crate is watermarked, and 0% of crate is watermarked
src_dir = Path('/n/fs/ncp/NCP.v2/data/images/carton_dugong/original/n02971356')
dst_dir.mkdir(parents=True, exist_ok=True)

# Watermark parameters
resize_size = 256
crop_size = 224
add_watermark = AddWatermark(image_size=crop_size, text="捷径捷径捷径")

transform_with_wm = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.CenterCrop(crop_size),
    #transforms.ToTensor(),
    add_watermark#,
    #transforms.ToPILImage()
])

transform_noop = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.CenterCrop(crop_size)
])

# --- Process ---

if binary_class == 'carton':
    all_imgs = sorted(list(src_dir.glob("*.jpeg")))  # assumes ImageNet naming
    subset_imgs = all_imgs[500:1000] #116
    num_to_watermark = int(wm_percent * len(subset_imgs))
    watermarked_imgs = random.sample(subset_imgs, num_to_watermark) ## debug random sample
elif binary_class == 'dugong_watermarked':
    '''
    subdirs = sorted([d for d in src_dir.iterdir() if d.is_dir()])
    subset_imgs = []
    watermarked_imgs = []
    for subdir in subdirs:
        imgs = sorted(subdir.glob("*.JPEG"))[500:616] # 167–204 inclusive
        subset_imgs.extend(imgs)  
        wm_count = int(wm_percent * len(imgs))
        watermarked_imgs.extend(imgs[:wm_count])
    '''
    # below if only one dir
    all_imgs = sorted(list(src_dir.glob("*.jpeg")))  # assumes ImageNet naming
    subset_imgs = all_imgs[500:1000]
    num_to_watermark = int(wm_percent * len(subset_imgs))
    watermarked_imgs = random.sample(subset_imgs, num_to_watermark) ## debug random sample
else: 
    raise RuntimeError('unknown binary class label')

print("wm_percent times len subset imgs = ", wm_percent * len(subset_imgs))
print("num watermarked images = ", len(watermarked_imgs))

for i, img_path in enumerate(subset_imgs):
    img = Image.open(img_path).convert("RGB")
    if img_path in watermarked_imgs:
        img = transform_with_wm(img)
    else:
        img = transform_noop(img)
    save_path = dst_dir / img_path.name
    img.save(save_path)
    print(f"Saved {'[WATERMARKED]' if img_path in watermarked_imgs else ''} {save_path}")
