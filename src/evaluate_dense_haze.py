import os
import torch
import lpips
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from src.models.unet import UNet
from src.models.unet_gated_cross_attention import UNetGatedCrossAttention


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# DENSE-HAZE PATHS
# -----------------------------
HAZY_DIR = "data/DENSE_HAZE/hazy"
GT_DIR = "data/DENSE_HAZE/gt"

B3_CKPT = "checkpoints/b3/best.pth"
B6_CKPT = "checkpoints/b6/best.pth"

# This is the B6 model fine-tuned on NH-HAZE.
# We are testing whether NH real-haze adaptation transfers to Dense-Haze.
B6_REAL_ADAPT_CKPT = "checkpoints/b6_nh/best.pth"

OUT_CSV = "results/dense_haze/metrics/dense_haze_b3_b6_b6realadapt_metrics.csv"

os.makedirs("results/dense_haze/metrics", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


def load_image(path):
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def load_b3():
    model = UNet().to(device)

    ckpt = torch.load(B3_CKPT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    print(f"Loaded B3 epoch {ckpt['epoch']}")
    return model


def load_b6(ckpt_path, label):
    model = UNetGatedCrossAttention(
        clip_model_name="ViT-B-16",
        clip_pretrained="openai",
        clip_input_size=224
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    model.clip_encoder.eval()

    print(f"Loaded {label} epoch {ckpt['epoch']}")
    return model


def evaluate_model(model, model_name, hazy_files, psnr_metric, ssim_metric, lpips_metric):
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    with torch.no_grad():
        for filename in tqdm(hazy_files, desc=f"Evaluating {model_name}"):
            hazy_path = os.path.join(HAZY_DIR, filename)

            gt_name = filename.replace("_hazy", "_GT")
            gt_path = os.path.join(GT_DIR, gt_name)

            if not os.path.exists(gt_path):
                print(f"Missing GT for {filename}, expected {gt_name}")
                continue

            hazy = load_image(hazy_path)
            gt = load_image(gt_path)

            output = model(hazy)

            psnr_scores.append(psnr_metric(output, gt).item())
            ssim_scores.append(ssim_metric(output, gt).item())
            lpips_scores.append(lpips_metric(output, gt).item())

    return {
        "model": model_name,
        "psnr": float(np.mean(psnr_scores)),
        "ssim": float(np.mean(ssim_scores)),
        "lpips": float(np.mean(lpips_scores)),
        "num_images": len(psnr_scores),
    }


def main():
    hazy_files = sorted([
        f for f in os.listdir(HAZY_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    print("DENSE-HAZE images:", len(hazy_files))

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    lpips_metric = lpips.LPIPS(net="alex").to(device)
    lpips_metric.eval()

    b3 = load_b3()
    b6 = load_b6(B6_CKPT, "B6")
    b6_real_adapt = load_b6(B6_REAL_ADAPT_CKPT, "B6_REAL_ADAPT_NH")

    results = []

    results.append(
        evaluate_model(
            b3,
            "B3",
            hazy_files,
            psnr_metric,
            ssim_metric,
            lpips_metric
        )
    )

    results.append(
        evaluate_model(
            b6,
            "B6",
            hazy_files,
            psnr_metric,
            ssim_metric,
            lpips_metric
        )
    )

    results.append(
        evaluate_model(
            b6_real_adapt,
            "B6_REAL_ADAPT_NH",
            hazy_files,
            psnr_metric,
            ssim_metric,
            lpips_metric
        )
    )

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)

    print("\n===== DENSE-HAZE Final Evaluation =====")
    print(df)


if __name__ == "__main__":
    main()