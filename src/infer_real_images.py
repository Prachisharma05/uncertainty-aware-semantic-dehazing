import os
from pathlib import Path

import yaml
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from src.models.unet import UNet
from src.models.unet_gated_cross_attention import UNetGatedCrossAttention


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_image(path, image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


def tensor_to_img(tensor):
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return img.clip(0, 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    input_dir = Path("data/real_hazy/rtts")
    output_b3_dir = Path("data/real_results/b3")
    output_b6_dir = Path("data/real_results/b6")
    comparison_dir = Path("data/real_results/comparisons")

    output_b3_dir.mkdir(parents=True, exist_ok=True)
    output_b6_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    b3_config = load_config("configs/b3_unet_clip_loss.yaml")
    b6_config = load_config("configs/b6_frozen_backbone.yaml")

    # Load B3
    b3_model = UNet(
        in_channels=b3_config["model"]["in_channels"],
        out_channels=b3_config["model"]["out_channels"]
    ).to(device)

    b3_ckpt = torch.load("checkpoints/b3/best.pth", map_location=device)
    b3_model.load_state_dict(b3_ckpt["model_state_dict"])
    b3_model.eval()

    print(f"Loaded B3 checkpoint epoch {b3_ckpt['epoch']}")

    # Load B6
    b6_model = UNetGatedCrossAttention(
        in_channels=b6_config["model"]["in_channels"],
        out_channels=b6_config["model"]["out_channels"],
        clip_model_name=b6_config["clip"]["model_name"],
        clip_pretrained=b6_config["clip"]["pretrained"],
        clip_input_size=b6_config["clip"]["input_size"]
    ).to(device)

    b6_ckpt = torch.load("checkpoints/b6/best.pth", map_location=device)
    b6_model.load_state_dict(b6_ckpt["model_state_dict"])
    b6_model.eval()
    b6_model.clip_encoder.eval()

    print(f"Loaded B6 checkpoint epoch {b6_ckpt['epoch']}")

    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    print(f"Found {len(image_paths)} real hazy images.")

    with torch.no_grad():
        for idx, img_path in enumerate(image_paths, start=1):
            hazy = load_image(img_path, image_size=256).to(device)

            b3_out = b3_model(hazy)
            b6_out, gate = b6_model(hazy, return_gate=True)

            save_image(b3_out, output_b3_dir / f"{img_path.stem}_b3.png")
            save_image(b6_out, output_b6_dir / f"{img_path.stem}_b6.png")

            gate_up = F.interpolate(
                gate,
                size=(256, 256),
                mode="bilinear",
                align_corners=False
            )

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            axes[0].imshow(tensor_to_img(hazy[0]))
            axes[0].set_title("Real Hazy")
            axes[0].axis("off")

            axes[1].imshow(tensor_to_img(b3_out[0]))
            axes[1].set_title("B3 Output")
            axes[1].axis("off")

            axes[2].imshow(tensor_to_img(b6_out[0]))
            axes[2].set_title("B6 Output")
            axes[2].axis("off")

            gate_map = gate_up[0, 0].detach().cpu().numpy()
            im = axes[3].imshow(gate_map, cmap="viridis")
            axes[3].set_title("B6 Gate Map")
            axes[3].axis("off")

            fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

            plt.tight_layout()
            save_path = comparison_dir / f"{img_path.stem}_comparison.png"
            plt.savefig(save_path, dpi=300)
            plt.close()

            print(f"[{idx}/{len(image_paths)}] Saved comparison: {save_path}")


if __name__ == "__main__":
    main()