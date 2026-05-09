import os
import yaml
import torch
import lpips
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

from src.datasets.dehaze_dataset import DehazeDataset
from src.models.unet import UNet


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def tensor_to_numpy(img_tensor):
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = img.clip(0, 1)
    return img


def evaluate():
    config = load_config("configs/b1_unet_l1.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    val_dataset = DehazeDataset(
        pairs_file=config["data"]["val_pairs"],
        image_size=config["data"]["image_size"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = UNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"]
    ).to(device)

    checkpoint_path = os.path.join(config["saving"]["checkpoint_dir"], "best.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation L1 loss: {checkpoint['val_loss']:.6f}")

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating B1"):
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)

            pred = model(hazy)

            pred_np = tensor_to_numpy(pred[0])
            clear_np = tensor_to_numpy(clear[0])

            psnr = peak_signal_noise_ratio(
                clear_np,
                pred_np,
                data_range=1.0
            )

            ssim = structural_similarity(
                clear_np,
                pred_np,
                channel_axis=2,
                data_range=1.0
            )

            # LPIPS expects [-1, 1]
            pred_lpips = pred * 2 - 1
            clear_lpips = clear * 2 - 1

            lpips_value = lpips_fn(pred_lpips, clear_lpips).item()

            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
            lpips_scores.append(lpips_value)

    results = {
        "model": "B1_UNet_L1",
        "checkpoint_epoch": checkpoint["epoch"],
        "val_l1_loss": checkpoint["val_loss"],
        "psnr": sum(psnr_scores) / len(psnr_scores),
        "ssim": sum(ssim_scores) / len(ssim_scores),
        "lpips": sum(lpips_scores) / len(lpips_scores),
    }

    os.makedirs("results/b1", exist_ok=True)

    df = pd.DataFrame([results])
    df.to_csv("results/b1/b1_metrics.csv", index=False)

    print("\n--- B1 Evaluation Results ---")
    print(df)


if __name__ == "__main__":
    evaluate()