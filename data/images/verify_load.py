from pathlib import Path

# This must match your earlier logic
src_dir = Path("imagenet_n03127925_binary/target")
dst_dir = Path("imagenet_n03127925_binary_prune_set/target_25_watermarked")

all_imgs = sorted(list(src_dir.glob("*.JPEG")))
subset_imgs = all_imgs[500:1000]
watermarked_set = set(img.name for img in subset_imgs[:125])  # Use names for filename matching

# Now count watermarked files in the destination dir
actual_imgs = sorted(list(dst_dir.glob("*.JPEG")))

watermarked_count = sum(1 for img in actual_imgs if img.name in watermarked_set)
non_watermarked_count = len(actual_imgs) - watermarked_count

print(f"Total files in directory: {len(actual_imgs)}")
print(f"Watermarked files: {watermarked_count}")
print(f"Non-watermarked files: {non_watermarked_count}")
