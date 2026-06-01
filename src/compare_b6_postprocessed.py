import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


B6_DIR = Path("data/real_results/b6")
B6_POST_DIR = Path("data/real_results/b6_postprocessed")
OUT_DIR = Path("data/real_results/b6_vs_postprocessed")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_rgb(path):
    img = cv2.imread(str(path))

    if img is None:
        raise ValueError(f"Could not read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def main():
    b6_images = sorted([
        p for p in B6_DIR.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    print(f"B6 images found: {len(b6_images)}")

    for idx, b6_path in enumerate(b6_images, start=1):
        post_path = B6_POST_DIR / b6_path.name

        if not post_path.exists():
            print(f"Missing postprocessed image for: {b6_path.name}")
            continue

        b6_img = read_rgb(b6_path)
        post_img = read_rgb(post_path)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(b6_img)
        axes[0].set_title("B6 Original")
        axes[0].axis("off")

        axes[1].imshow(post_img)
        axes[1].set_title("B6 Postprocessed")
        axes[1].axis("off")

        plt.tight_layout()

        save_path = OUT_DIR / f"{b6_path.stem}_compare.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"[{idx}/{len(b6_images)}] Saved: {save_path}")


if __name__ == "__main__":
    main()