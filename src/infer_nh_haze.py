import os
import torch
import lpips
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from src.models.unet import UNet
from src.models.unet_gated_cross_attention import UNetGatedCrossAttention


# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# PATHS
# -----------------------------
HAZY_DIR = "data/NH_HAZE/hazy"
GT_DIR = "data/NH_HAZE/gt"

B3_CKPT = "checkpoints/b3/best.pth"
B6_CKPT = "checkpoints/b6/best.pth"

OUT_B3 = "results/nh_haze/b3"
OUT_B6 = "results/nh_haze/b6"

os.makedirs(OUT_B3, exist_ok=True)
os.makedirs(OUT_B6, exist_ok=True)


# -----------------------------
# IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])


# -----------------------------
# LOAD MODELS
# -----------------------------
b3_model = UNet().to(device)

b3_ckpt = torch.load(B3_CKPT, map_location=device)
b3_model.load_state_dict(b3_ckpt["model_state_dict"])
b3_model.eval()


b6_model = UNetGatedCrossAttention().to(device)

b6_ckpt = torch.load(B6_CKPT, map_location=device)
b6_model.load_state_dict(b6_ckpt["model_state_dict"])
b6_model.eval()


# -----------------------------
# METRICS
# -----------------------------
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

lpips_metric = lpips.LPIPS(net='alex').to(device)


# -----------------------------
# HELPERS
# -----------------------------
def load_image(path):
    image = Image.open(path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


def save_tensor_image(tensor, path):
    image = tensor.squeeze(0).detach().cpu().clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


# -----------------------------
# EVALUATION STORAGE
# -----------------------------
b3_psnr = []
b3_ssim = []
b3_lpips = []

b6_psnr = []
b6_ssim = []
b6_lpips = []


# -----------------------------
# INFERENCE LOOP
# -----------------------------
hazy_files = sorted(os.listdir(HAZY_DIR))

for filename in tqdm(hazy_files):

    hazy_path = os.path.join(HAZY_DIR, filename)

    gt_name = filename.replace("_hazy", "_GT")
    gt_path = os.path.join(GT_DIR, gt_name)

    hazy = load_image(hazy_path)
    gt = load_image(gt_path)

    with torch.no_grad():

        # -----------------------------
        # B3
        # -----------------------------
        b3_out = b3_model(hazy)

        # -----------------------------
        # B6
        # -----------------------------
        b6_out = b6_model(hazy)

    # -----------------------------
    # SAVE OUTPUTS
    # -----------------------------
    save_tensor_image(
        b3_out,
        os.path.join(OUT_B3, filename)
    )

    save_tensor_image(
        b6_out,
        os.path.join(OUT_B6, filename)
    )

    # -----------------------------
    # B3 METRICS
    # -----------------------------
    b3_psnr.append(
        psnr_metric(b3_out, gt).item()
    )

    b3_ssim.append(
        ssim_metric(b3_out, gt).item()
    )

    b3_lpips.append(
        lpips_metric(b3_out, gt).item()
    )

    # -----------------------------
    # B6 METRICS
    # -----------------------------
    b6_psnr.append(
        psnr_metric(b6_out, gt).item()
    )

    b6_ssim.append(
        ssim_metric(b6_out, gt).item()
    )

    b6_lpips.append(
        lpips_metric(b6_out, gt).item()
    )


# -----------------------------
# FINAL RESULTS
# -----------------------------
print("\n===== NH-HAZE RESULTS =====")

print("\nB3 Results:")
print(f"PSNR  : {np.mean(b3_psnr):.4f}")
print(f"SSIM  : {np.mean(b3_ssim):.4f}")
print(f"LPIPS : {np.mean(b3_lpips):.4f}")

print("\nB6 Results:")
print(f"PSNR  : {np.mean(b6_psnr):.4f}")
print(f"SSIM  : {np.mean(b6_ssim):.4f}")
print(f"LPIPS : {np.mean(b6_lpips):.4f}")