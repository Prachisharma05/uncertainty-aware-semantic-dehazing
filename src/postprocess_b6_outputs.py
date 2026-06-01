import os
import cv2
import numpy as np
from glob import glob

INPUT_DIR = "data/real_results/b6"
OUTPUT_DIR = "data/real_results/b6_postprocessed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_paths = sorted(glob(os.path.join(INPUT_DIR, "*.png")))

def gray_world_correction(img):
    img = img.astype(np.float32)

    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])

    avg_gray = (avg_b + avg_g + avg_r) / 3

    img[:, :, 0] *= avg_gray / (avg_b + 1e-6)
    img[:, :, 1] *= avg_gray / (avg_g + 1e-6)
    img[:, :, 2] *= avg_gray / (avg_r + 1e-6)

    img = np.clip(img, 0, 255)

    return img.astype(np.uint8)

def gamma_correction(img, gamma=1.15):
    img = img.astype(np.float32) / 255.0
    img = np.power(img, gamma)
    img = np.clip(img * 255.0, 0, 255)

    return img.astype(np.uint8)

def contrast_normalization(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

for path in image_paths:

    img = cv2.imread(path)

    # Step 1: color correction
    img = gray_world_correction(img)

    # Step 2: gamma stabilization
    img = gamma_correction(img, gamma=1.1)

    # Step 3: local contrast enhancement
    img = contrast_normalization(img)

    name = os.path.basename(path)

    save_path = os.path.join(OUTPUT_DIR, name)

    cv2.imwrite(save_path, img)

    print(f"Saved: {save_path}")

print("\nPost-processing complete.")