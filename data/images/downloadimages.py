import os
import requests
import random
from pathlib import Path
from tqdm import tqdm

# Constants
target_wnid = 'n02802426' #https://github.com/formigone/tf-imagenet/blob/master/LOC_synset_mapping.txt
num_non_target = 1000
output_dir = Path(f'./imagenet_{target_wnid}_binary')
output_dir.mkdir(exist_ok=True)

def fetch_urls(wnid):
    url = f"http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={wnid}"
    resp = requests.get(url)
    return resp.text.strip().split('\n')

def download_images(urls, dest_dir, limit=None):
    os.makedirs(dest_dir, exist_ok=True)
    downloaded = 0
    for idx, url in enumerate(tqdm(urls)):
        try:
            img_data = requests.get(url.strip(), timeout=5).content
            with open(os.path.join(dest_dir, f"{idx}.jpg"), 'wb') as f:
                f.write(img_data)
            downloaded += 1
            if limit and downloaded >= limit:
                break
        except Exception:
            continue

# Step 1: Download target images
target_urls = fetch_urls(target_wnid)
download_images(target_urls, output_dir / f'{target_wnid}', limit=1500)

# Step 2: Download non-target images from random synsets
all_synsets = [line.strip() for line in requests.get(
    "https://raw.githubusercontent.com/fab-jul/ImageNetV2/main/synset_words.txt"
).text.splitlines()]
non_target_synsets = [s.split()[0] for s in all_synsets if s.split()[0] != target_wnid]

random.shuffle(non_target_synsets)

nontarget_dir = output_dir / 'not_basketball'
nontarget_dir.mkdir(exist_ok=True)

count = 0
for syn in non_target_synsets:
    urls = fetch_urls(syn)
    pre_count = count
    download_images(urls, nontarget_dir, limit=5)  # try up to 5 from each
    count = len(os.listdir(nontarget_dir))
    if count >= num_non_target:
        break
print(f"Total non-target images: {count}")
