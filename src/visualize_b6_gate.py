import os
import yaml
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.dehaze_dataset import DehazeDataset
from src.models.unet import UNet
from src.models.unet_gated_cross_attention import UNetGatedCrossAttention


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def tensor_to_img(tensor):
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return img.clip(0, 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    b3_config = load_config("configs/b3_unet_clip_loss.yaml")
    b6_config = load_config("configs/b6_frozen_backbone.yaml")

    val_dataset = DehazeDataset(
        pairs_file=b6_config["data"]["val_pairs"],
        image_size=b6_config["data"]["image_size"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    # Load B3
    b3_model = UNet(
        in_channels=b3_config["model"]["in_channels"],
        out_channels=b3_config["model"]["out_channels"]
    ).to(device)

    b3_ckpt = torch.load("checkpoints/b3/best.pth", map_location=device)
    b3_model.load_state_dict(b3_ckpt["model_state_dict"])
    b3_model.eval()

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

    batch = next(iter(val_loader))

    hazy = batch["hazy"].to(device)
    clear = batch["clear"].to(device)

    with torch.no_grad():
        b3_out = b3_model(hazy)
        b6_out, gate = b6_model(hazy, return_gate=True)

    gate_up = F.interpolate(
        gate,
        size=(256, 256),
        mode="bilinear",
        align_corners=False
    )

    os.makedirs("results/phase8_visuals", exist_ok=True)

    for i in range(4):
        fig, axes = plt.subplots(1, 6, figsize=(22, 4))

        axes[0].imshow(tensor_to_img(hazy[i]))
        axes[0].set_title("Hazy Input")
        axes[0].axis("off")

        axes[1].imshow(tensor_to_img(b3_out[i]))
        axes[1].set_title("B3 Output")
        axes[1].axis("off")

        axes[2].imshow(tensor_to_img(b6_out[i]))
        axes[2].set_title("B6 Output")
        axes[2].axis("off")

        axes[3].imshow(tensor_to_img(clear[i]))
        axes[3].set_title("Ground Truth")
        axes[3].axis("off")

        gate_map = gate_up[i, 0].detach().cpu().numpy()
        im = axes[4].imshow(gate_map, cmap="viridis")
        axes[4].set_title("B6 Gate Map")
        axes[4].axis("off")
        
        axes[5].imshow(tensor_to_img(hazy[i]))
        im2 = axes[5].imshow(gate_map, cmap="jet", alpha=0.5)
        axes[5].set_title("Gate Overlay (Hazy)")
        axes[5].axis("off")

        fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[5], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = f"results/phase8_visuals/b6_gate_analysis_{i}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()