from pathlib import Path


HAZY_DIR = Path("data/NH_HAZE/hazy")
GT_DIR = Path("data/NH_HAZE/gt")

OUT_FILE = Path("data/NH_HAZE/nh_pairs.txt")

valid_exts = {".png", ".jpg", ".jpeg"}

hazy_files = sorted([
    p for p in HAZY_DIR.iterdir()
    if p.suffix.lower() in valid_exts
])

pairs = []

for hazy_path in hazy_files:
    gt_name = hazy_path.name.replace("_hazy", "_GT")
    gt_path = GT_DIR / gt_name

    if gt_path.exists():
        pairs.append((hazy_path, gt_path))
    else:
        print(f"Missing GT for: {hazy_path.name}")

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_FILE, "w") as f:
    for hazy, gt in pairs:
        f.write(f"{hazy.as_posix()} {gt.as_posix()}\n")

print("NH-HAZE pair file created.")
print("Total pairs:", len(pairs))
print("Saved to:", OUT_FILE)