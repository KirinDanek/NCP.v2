import os, io, tarfile, urllib.request
from pathlib import Path

TARGET_SYNSET = ""  
# n02971356 # carton
# n02074367 #dugong
BASE_OUTPUT_DIR = Path("/n/fs/ncp/NCP.v2/data/images/carton_dugong/")
BASE_URL = "https://image-net.org/data/winter21_whole"  # per-synset tarballs live here
NUM_PER_BINARY_CLASS = 999  


import ssl, certifi, urllib.request

# Point urllib at the certifi CA bundle
ctx = ssl.create_default_context(cafile=certifi.where())
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
opener.addheaders = [("User-Agent", "Mozilla/5.0 (wget-like)")]
urllib.request.install_opener(opener)

def download_tar(wnid: str, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_url = f"{BASE_URL}/{wnid}.tar"
    tar_bytes = urllib.request.urlopen(tar_url, timeout=30).read()  # uses our opener/context
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tar:
        members = [m for m in tar.getmembers() if m.name.lower().endswith((".jpeg", ".jpg"))]
        if not members:
            raise RuntimeError(f"No JPEGs found inside {wnid}.tar (URL: {tar_url})")
        if len(members) < NUM_PER_BINARY_CLASS:
            print(f"[warn] only {len(members)} images in {wnid}.tar")

        kept = 0
        for m in members:
            f = tar.extractfile(m)
            if f is None:
                continue
            # write to flat dir with unique filename
            out_path = dest_dir / f"{wnid}_{kept:05d}.jpeg"
            with open(out_path, "wb") as o:
                o.write(f.read())
            kept += 1
        print(f"[ok] Extracted {kept} images to {dest_dir}")

# ----- run for carton -----
print(f"processing target {TARGET_SYNSET}")
try:
    out_dir = BASE_OUTPUT_DIR / TARGET_SYNSET
    download_tar(TARGET_SYNSET, out_dir)
except Exception as e:
    print(f"[fail] {TARGET_SYNSET}: {e}")
