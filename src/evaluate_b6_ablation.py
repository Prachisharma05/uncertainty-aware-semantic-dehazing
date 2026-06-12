import os
import yaml
import torch
import lpips
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

from src.datasets.dehaze_dataset import DehazeDataset
from src.models.unet_gated_cross_attention import UNetGatedCrossAttention


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def tensor_to_numpy(img_tensor):
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    return img.clip(0, 1)


def evaluate_ablation():
    config = load_config("configs/b6_frozen_backbone.yaml")

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

    model = UNetGatedCrossAttention(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        clip_model_name=config["clip"]["model_name"],
        clip_pretrained=config["clip"]["pretrained"],
        clip_input_size=config["clip"]["input_size"]
    ).to(device)

    checkpoint_path = os.path.join(config["saving"]["checkpoint_dir"], "best.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.clip_encoder.eval()

    print(f"Loaded B6 checkpoint from epoch {checkpoint['epoch']}")

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    gate_scores = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating B6 Ablation"):
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)

            # Pass ablate_semantics=True
            pred, gate = model(hazy, return_gate=True, ablate_semantics=True)

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

            pred_lpips = pred * 2 - 1
            clear_lpips = clear * 2 - 1

            lpips_value = lpips_fn(pred_lpips, clear_lpips).item()

            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
            lpips_scores.append(lpips_value)
            gate_scores.append(gate.mean().item())

    results = {
        "model": "B6_FrozenBackbone_Ablation",
        "checkpoint_epoch": checkpoint["epoch"],
        "psnr": sum(psnr_scores) / len(psnr_scores),
        "ssim": sum(ssim_scores) / len(ssim_scores),
        "lpips": sum(lpips_scores) / len(lpips_scores),
        "eval_gate_mean": sum(gate_scores) / len(gate_scores),
    }

    os.makedirs("results/b6", exist_ok=True)

    df = pd.DataFrame([results])
    df.to_csv("results/b6/b6_ablation_metrics.csv", index=False)

    print("\n--- B6 Ablation Evaluation Results ---")
    print(df)


if __name__ == "__main__":
    evaluate_ablation()
