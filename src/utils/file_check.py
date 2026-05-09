from pathlib import Path


def check_dataset():
    hazy_dir = Path("data/RESIDE/ITS/hazy")
    clear_dir = Path("data/RESIDE/ITS/clear")

    hazy_files = sorted(list(hazy_dir.glob("*.*")))
    clear_files = set([p.name for p in clear_dir.glob("*.*")])

    total = len(hazy_files)
    missing = []
    checked = 0

    for hazy_path in hazy_files:
        clean_id = hazy_path.stem.split("_")[0]
        clean_name = clean_id + hazy_path.suffix

        if clean_name not in clear_files:
            missing.append((hazy_path.name, clean_name))

        checked += 1

        if checked % 2000 == 0:
            print(f"Checked {checked}/{total}...")

    print("\n--- DATASET CHECK REPORT ---")
    print(f"Total hazy images: {total}")
    print(f"Missing clean pairs: {len(missing)}")

    if len(missing) > 0:
        print("\nExamples of mismatches:")
        for m in missing[:10]:
            print(m)
    else:
        print("All hazy images have valid clean pairs.")


if __name__ == "__main__":
    check_dataset()