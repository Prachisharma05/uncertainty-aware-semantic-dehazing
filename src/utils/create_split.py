from pathlib import Path
import random


def create_train_val_split(
    hazy_dir="data/RESIDE/ITS/hazy",
    clear_dir="data/RESIDE/ITS/clear",
    train_ratio=0.9,
    seed=42
):
    hazy_dir = Path(hazy_dir)
    clear_dir = Path(clear_dir)

    hazy_files = sorted(list(hazy_dir.glob("*.*")))

    pairs = []

    for hazy_path in hazy_files:
        clean_id = hazy_path.stem.split("_")[0]
        clear_path = clear_dir / f"{clean_id}{hazy_path.suffix}"

        if clear_path.exists():
            pairs.append((str(hazy_path), str(clear_path)))

    random.seed(seed)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * train_ratio)

    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    train_file = Path("data/RESIDE/ITS/train_pairs.txt")
    val_file = Path("data/RESIDE/ITS/val_pairs.txt")

    with open(train_file, "w") as f:
        for hazy, clear in train_pairs:
            f.write(f"{hazy},{clear}\n")

    with open(val_file, "w") as f:
        for hazy, clear in val_pairs:
            f.write(f"{hazy},{clear}\n")

    print("Train/validation split created.")
    print(f"Total pairs: {len(pairs)}")
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    print(f"Train file: {train_file}")
    print(f"Validation file: {val_file}")


if __name__ == "__main__":
    create_train_val_split()