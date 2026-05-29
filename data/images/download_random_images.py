import os
import random
import tarfile
import urllib.request
from pathlib import Path

# ---- CONFIG ----
OUTPUT_DIR = Path("imagenet_430_binary/not_basketball")
BASKETBALL_SYNSET = "n02802426"
SYNSET_LIST_URL = "https://image-net.org/api/wordpress/?synsetids=true"  # fallback if you need one
BASE_URL = "https://image-net.org/data/winter21_whole"
NUM_CLASSES = 110
NUM_IMAGES_PER_CLASS = 10

# ---- Setup ----
os.makedirs(OUTPUT_DIR, exist_ok=True)

# You need a list of 1000+ valid synset IDs
# Option A: hardcoded from download page or .txt file
# Option B: load from local file or ImageNet’s public list
with open("imagenet_synsets.txt", "r") as f:
    all_synsets = [line.strip() for line in f if line.strip() != BASKETBALL_SYNSET]

# Sample 100 random synsets
random.seed(42)
chosen_synsets = random.sample(all_synsets, NUM_CLASSES)

# ---- Download Loop ----
for synset in chosen_synsets:
    print(f"Processing synset {synset}...")
    tar_url = f"{BASE_URL}/{synset}.tar"
    tar_path = f"{synset}.tar"
    try:
        # download .tar
        urllib.request.urlretrieve(tar_url, tar_path)

        # extract 10 files only
        with tarfile.open(tar_path) as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".JPEG")]
            if len(members) < NUM_IMAGES_PER_CLASS:
                print(f"  Skipping {synset}: only {len(members)} images")
                continue
            dest_dir = OUTPUT_DIR / synset
            dest_dir.mkdir(parents=True, exist_ok=True)
            for m in members[:NUM_IMAGES_PER_CLASS]:
                tar.extract(m, path=dest_dir)

        # cleanup
        os.remove(tar_path)

    except Exception as e:
        print(f"  Failed to process {synset}: {e}")
        if os.path.exists(tar_path):
            os.remove(tar_path)
