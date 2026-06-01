import os
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from src.models.unet_gated_cross_attention import UNetGatedCrossAttention
from src.utils.seed import set_seed


class NHHazeDataset(Dataset):
    def __init__(self, pair_file, image_size=256):
        self.image_size = image_size

        with open(pair_file, "r") as f:
            self.pairs = [line.strip().split() for line in f.readlines()]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hazy_path, gt_path = self.pairs[idx]

        hazy = Image.open(hazy_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")

        hazy = self.transform(hazy)
        gt = self.transform(gt)

        return {
            "hazy": hazy,
            "clear": gt
        }


def load_config(path):
    with open(path, "r") as f:
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


def compute_clip_loss(pred, clear, clip_model, clip_input_size):
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

    return 1 - F.cosine_similarity(pred_feat, clear_feat, dim=-1).mean()


def compute_loss(pred, clear, gate, l1_fn, lpips_fn, clip_model, config):
    l1 = l1_fn(pred, clear)

    pred_lpips = pred * 2 - 1
    clear_lpips = clear * 2 - 1

    perceptual = lpips_fn(pred_lpips, clear_lpips).mean()

    semantic = compute_clip_loss(
        pred,
        clear,
        clip_model,
        config["clip"]["input_size"]
    )

    gate_reg = (gate.mean() - 0.5) ** 2

    total = (
        config["loss"]["lambda_l1"] * l1
        + config["loss"]["lambda_lpips"] * perceptual
        + config["loss"]["lambda_clip"] * semantic
        + config["loss"]["lambda_gate"] * gate_reg
    )

    return total, l1.item(), perceptual.item(), semantic.item(), gate.mean().item()


def freeze_for_nh_finetune(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.gated_cross_attention.parameters():
        param.requires_grad = True

    model.clip_encoder.eval()

    for param in model.clip_encoder.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Frozen B6 backbone and CLIP encoder.")
    print("Training only gated_cross_attention.")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable percent: {100 * trainable_params / total_params:.2f}%")


def validate(model, loader, l1_fn, lpips_fn, clip_model, config, device):
    model.eval()
    model.clip_encoder.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_lpips = 0.0
    total_clip = 0.0
    total_gate = 0.0

    with torch.no_grad():
        for batch in loader:
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)

            pred, gate = model(hazy, return_gate=True)

            loss, l1, perceptual, semantic, gate_mean = compute_loss(
                pred,
                clear,
                gate,
                l1_fn,
                lpips_fn,
                clip_model,
                config
            )

            total_loss += loss.item()
            total_l1 += l1
            total_lpips += perceptual
            total_clip += semantic
            total_gate += gate_mean

    n = len(loader)

    return (
        total_loss / n,
        total_l1 / n,
        total_lpips / n,
        total_clip / n,
        total_gate / n
    )


def save_samples(model, loader, device, result_dir, epoch):
    model.eval()
    model.clip_encoder.eval()

    batch = next(iter(loader))
    hazy = batch["hazy"].to(device)
    clear = batch["clear"].to(device)

    with torch.no_grad():
        pred, gate = model(hazy, return_gate=True)

    comparison = torch.cat(
        [hazy[:4], pred[:4], clear[:4]],
        dim=0
    )

    save_image(
        comparison,
        os.path.join(result_dir, f"epoch_{epoch:03d}_samples.png"),
        nrow=4
    )


def train():
    config = load_config("configs/b6_nh_finetune.yaml")

    set_seed(config["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    save_dir = config["checkpoint"]["save_dir"]
    result_dir = config["results"]["save_dir"]

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    train_dataset = NHHazeDataset(
        pair_file=config["data"]["train_pairs"],
        image_size=config["data"]["image_size"]
    )

    val_dataset = NHHazeDataset(
        pair_file=config["data"]["val_pairs"],
        image_size=config["data"]["image_size"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )

    model = UNetGatedCrossAttention(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        clip_model_name=config["clip"]["model_name"],
        clip_pretrained=config["clip"]["pretrained"],
        clip_input_size=config["clip"]["input_size"]
    ).to(device)

    b6_ckpt = torch.load("checkpoints/b6/best.pth", map_location=device)
    model.load_state_dict(b6_ckpt["model_state_dict"])
    print(f"Loaded B6 checkpoint epoch {b6_ckpt['epoch']}")

    freeze_for_nh_finetune(model)

    l1_fn = nn.L1Loss()

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    for param in lpips_fn.parameters():
        param.requires_grad = False

    clip_model = model.clip_encoder.clip_model
    clip_model.eval()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["optimizer"]["lr"],
        weight_decay=1e-5
    )

    best_val_loss = float("inf")

    log_path = os.path.join(result_dir, "train_log.csv")

    with open(log_path, "w") as f:
        f.write(
            "epoch,train_total,train_l1,train_lpips,train_clip,train_gate,"
            "val_total,val_l1,val_lpips,val_clip,val_gate\n"
        )

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        model.clip_encoder.eval()

        total_loss = 0.0
        total_l1 = 0.0
        total_lpips = 0.0
        total_clip = 0.0
        total_gate = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config['training']['epochs']}"
        )

        for batch in progress:
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)

            optimizer.zero_grad()

            pred, gate = model(hazy, return_gate=True)

            loss, l1, perceptual, semantic, gate_mean = compute_loss(
                pred,
                clear,
                gate,
                l1_fn,
                lpips_fn,
                clip_model,
                config
            )

            if torch.isnan(loss):
                print("NaN detected. Stopping.")
                return

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.gated_cross_attention.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()
            total_l1 += l1
            total_lpips += perceptual
            total_clip += semantic
            total_gate += gate_mean

            progress.set_postfix({
                "loss": loss.item(),
                "l1": l1,
                "lpips": perceptual,
                "clip": semantic,
                "gate": gate_mean
            })

        n = len(train_loader)

        train_total = total_loss / n
        train_l1 = total_l1 / n
        train_lpips = total_lpips / n
        train_clip = total_clip / n
        train_gate = total_gate / n

        val_total, val_l1, val_lpips, val_clip, val_gate = validate(
            model,
            val_loader,
            l1_fn,
            lpips_fn,
            clip_model,
            config,
            device
        )

        print(
            f"Epoch [{epoch}/{config['training']['epochs']}] "
            f"Train Total: {train_total:.6f} | "
            f"Train L1: {train_l1:.6f} | "
            f"Train LPIPS: {train_lpips:.6f} | "
            f"Train CLIP: {train_clip:.6f} | "
            f"Train Gate: {train_gate:.6f} | "
            f"Val Total: {val_total:.6f} | "
            f"Val L1: {val_l1:.6f} | "
            f"Val LPIPS: {val_lpips:.6f} | "
            f"Val CLIP: {val_clip:.6f} | "
            f"Val Gate: {val_gate:.6f}"
        )

        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{train_total},{train_l1},{train_lpips},{train_clip},{train_gate},"
                f"{val_total},{val_l1},{val_lpips},{val_clip},{val_gate}\n"
            )

        save_samples(model, val_loader, device, result_dir, epoch)

        last_path = os.path.join(save_dir, "last.pth")
        best_path = os.path.join(save_dir, "best.pth")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_total_loss": val_total,
                "val_l1": val_l1,
                "val_lpips": val_lpips,
                "val_clip": val_clip,
                "val_gate": val_gate,
            },
            last_path
        )

        if val_total < best_val_loss:
            best_val_loss = val_total

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_total_loss": val_total,
                    "val_l1": val_l1,
                    "val_lpips": val_lpips,
                    "val_clip": val_clip,
                    "val_gate": val_gate,
                },
                best_path
            )

            print(
                f"Best B6_NH model saved at epoch {epoch} "
                f"with val loss {best_val_loss:.6f}"
            )


if __name__ == "__main__":
    train()