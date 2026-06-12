import os
import yaml
import torch
import torch.nn.functional as F
import open_clip
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.datasets.dehaze_dataset import DehazeDataset
from src.models.unet import UNet
from src.models.unet_gated_cross_attention import UNetGatedCrossAttention


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def clip_normalize(img):
    mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073],
        device=img.device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711],
        device=img.device
    ).view(1, 3, 1, 1)

    return (img - mean) / std


def compute_cosine_similarity(pred, clear, clip_model, clip_input_size):
    pred_resized = F.interpolate(
        pred,
        size=(clip_input_size, clip_input_size),
        mode="bilinear",
        align_corners=False
    )

    clear_resized = F.interpolate(
        clear,
        size=(clip_input_size, clip_input_size),
        mode="bilinear",
        align_corners=False
    )

    pred_norm = clip_normalize(pred_resized)
    clear_norm = clip_normalize(clear_resized)

    pred_feat = clip_model.encode_image(pred_norm)
    clear_feat = clip_model.encode_image(clear_norm)

    pred_feat = pred_feat / pred_feat.norm(dim=-1, keepdim=True)
    clear_feat = clear_feat / clear_feat.norm(dim=-1, keepdim=True)

    return F.cosine_similarity(pred_feat, clear_feat, dim=-1).mean().item()


def evaluate_similarity():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    b3_config = load_config("configs/b3_unet_clip_loss.yaml")
    b6_config = load_config("configs/b6_frozen_backbone.yaml")

    val_dataset = DehazeDataset(
        pairs_file=b6_config["data"]["val_pairs"],
        image_size=b6_config["data"]["image_size"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # OpenCLIP Model
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="openai"
    )
    clip_model = clip_model.to(device)
    clip_model.eval()

    # Load B1 (from B3 config but we'll use b1 checkpoint and UNet)
    b1_model = UNet(
        in_channels=b3_config["model"]["in_channels"],
        out_channels=b3_config["model"]["out_channels"]
    ).to(device)
    b1_ckpt = torch.load("checkpoints/b1/best.pth", map_location=device)
    b1_model.load_state_dict(b1_ckpt["model_state_dict"])
    b1_model.eval()

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

    b1_scores = []
    b6_scores = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating Semantic Similarity"):
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)

            b1_pred = b1_model(hazy)
            b6_pred, _ = b6_model(hazy, return_gate=True)

            b1_sim = compute_cosine_similarity(b1_pred, clear, clip_model, 224)
            b6_sim = compute_cosine_similarity(b6_pred, clear, clip_model, 224)

            b1_scores.append(b1_sim)
            b6_scores.append(b6_sim)

    avg_b1 = sum(b1_scores) / len(b1_scores)
    avg_b6 = sum(b6_scores) / len(b6_scores)

    results = [
        {"Model": "B1_Baseline", "CLIP_Cosine_Similarity": avg_b1},
        {"Model": "B6_Gated_Semantic", "CLIP_Cosine_Similarity": avg_b6}
    ]

    os.makedirs("results/final", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/final/semantic_similarity.csv", index=False)

    print("\n--- Semantic Similarity Results ---")
    print(df)


if __name__ == "__main__":
    evaluate_similarity()
