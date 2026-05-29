CUB=/n/fs/ncp/NCP.v2/data/images/CUB
OUT=/n/fs/ncp/NCP.v2/data/images/waterbirds_out
mkdir -p "$OUT"

python3 - <<'PY'
import os, pandas as pd, numpy as np, random
random.seed(42)
CUB=os.environ['CUB']

# Load CUB metadata
images = pd.read_csv(os.path.join(CUB,'images.txt'), sep=' ', header=None, names=['id','rel'])
splits = pd.read_csv(os.path.join(CUB,'train_test_split.txt'), sep=' ', header=None, names=['id','split'])
df = images.merge(splits, on='id')
df = df[df['split']==1]     # original CUB train only

# Waterbird label (same heuristic used by generate_waterbirds.py)
water_words=set(s.lower() for s in [
  'Albatross','Auklet','Cormorant','Frigatebird','Fulmar','Gull','Jaeger',
  'Kittiwake','Pelican','Puffin','Tern','Gadwall','Grebe','Mallard',
  'Merganser','Guillemot','Pacific_Loon'
])
def is_water(rel):
    spec = rel.split('/')[0].split('.')[1].lower()
    return int(any(w in spec for w in water_words))

df['y'] = df['rel'].apply(is_water)
nW = (df['y']==1).sum()
nL = (df['y']==0).sum()
n = min(nW, nL)  # target per class for perfect 50/50

w = df[df['y']==1].sample(n=n, random_state=42)
l = df[df['y']==0].sample(n=n, random_state=42)
sel = pd.concat([w,l]).sample(frac=1.0, random_state=42)

out = os.path.join(os.environ['OUT'], 'cub_whitelist_A_balanced.txt')
sel['rel'].to_csv(out, index=False, header=False)
print(f"Wrote {out} with total={len(sel)} (water={len(w)}, land={len(l)})")
PY
