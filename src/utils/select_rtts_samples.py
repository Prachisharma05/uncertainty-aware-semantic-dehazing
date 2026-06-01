from pathlib import Path
import shutil
from PIL import Image
import random


SOURCE_DIR = Path("data/raw_rtts/JPEGImages")   # change if your RTTS is elsewhere
TARGET_DIR = Path("data/real_hazy/rtts")

TARGET_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

valid_exts = {".jpg", ".jpeg", ".png"}

image_paths = [
    p for p in SOURCE_DIR.iterdir()
    if p.suffix.lower() in valid_exts
]

print("Total RTTS images found:", len(image_paths))

valid_images = []

for p in image_paths:
    try:
        img = Image.open(p).convert("RGB")
        w, h = img.size

        if w >= 300 and h >= 300:
            valid_images.append((p, w, h))

    except Exception as e:
        print("Skipping broken image:", p.name, e)

print("Valid images:", len(valid_images))

# Shuffle and select 20 images
random.shuffle(valid_images)
selected = valid_images[:20]

for idx, (src, w, h) in enumerate(selected, start=1):
    new_name = f"rtts_{idx:02d}{src.suffix.lower()}"
    dst = TARGET_DIR / new_name
    shutil.copy(src, dst)
    print(f"Copied {src.name} -> {new_name} | size: {w}x{h}")

print("\nDone. Selected images saved to:")
print(TARGET_DIR)