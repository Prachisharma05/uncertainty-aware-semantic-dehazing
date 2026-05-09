import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
import open_clip
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.datasets.dehaze_dataset import DehazeDataset
from src.models.unet_cross_attention import UNetCrossAttention
from src.utils.seed import set_seed


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

    clip_loss = 1 - F.cosine_similarity(pred_feat, clear_feat, dim=-1).mean()

    return clip_loss


def compute_loss(pred, clear, l1_fn, lpips_fn, clip_model, config):
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

    total = (
        config["loss"]["lambda_l1"] * l1
        + config["loss"]["lambda_lpips"] * perceptual
        + config["loss"]["lambda_clip"] * semantic
    )

    return total, l1.item(), perceptual.item(), semantic.item()


def validate(model, val_loader, l1_fn, lpips_fn, clip_model, config, device):
    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_lpips = 0.0
    total_clip = 0.0

    with torch.no_grad():
        for batch in val_loader:
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)

            pred = model(hazy)

            loss, l1_value, lpips_value, clip_value = compute_loss(
                pred,
                clear,
                l1_fn,
                lpips_fn,
                clip_model,
                config
            )

            total_loss += loss.item()
            total_l1 += l1_value
            total_lpips += lpips_value
            total_clip += clip_value

    n = len(val_loader)

    return (
        total_loss / n,
        total_l1 / n,
        total_lpips / n,
        total_clip / n
    )


def save_samples(model, val_loader, device, result_dir, epoch):
    model.eval()

    batch = next(iter(val_loader))
    hazy = batch["hazy"].to(device)
    clear = batch["clear"].to(device)

    with torch.no_grad():
        pred = model(hazy)

    comparison = torch.cat(
        [hazy[:4], pred[:4], clear[:4]],
        dim=0
    )

    save_path = os.path.join(result_dir, f"epoch_{epoch:03d}_samples.png")
    save_image(comparison, save_path, nrow=4)


def load_partial_b3_weights(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print("B3 checkpoint not found. Training B4 from scratch.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    b3_state = checkpoint["model_state_dict"]
    b4_state = model.state_dict()

    loaded_keys = []
    skipped_keys = []

    for key, value in b3_state.items():
        if key in b4_state and b4_state[key].shape == value.shape:
            b4_state[key] = value
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    model.load_state_dict(b4_state)

    print(f"Initialized B4 partially from B3 checkpoint epoch {checkpoint['epoch']}")
    print(f"Loaded compatible keys: {len(loaded_keys)}")
    print(f"Skipped keys: {len(skipped_keys)}")


def train():
    config = load_config("configs/b4_cross_attention.yaml")

    set_seed(config["experiment"]["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    os.makedirs(config["saving"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["saving"]["result_dir"], exist_ok=True)
    os.makedirs(config["saving"]["log_dir"], exist_ok=True)

    train_dataset = DehazeDataset(
        pairs_file=config["data"]["train_pairs"],
        image_size=config["data"]["image_size"]
    )

    val_dataset = DehazeDataset(
        pairs_file=config["data"]["val_pairs"],
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

    model = UNetCrossAttention(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        clip_model_name=config["clip"]["model_name"],
        clip_pretrained=config["clip"]["pretrained"],
        clip_input_size=config["clip"]["input_size"]
    ).to(device)

    load_partial_b3_weights(
        model=model,
        checkpoint_path="checkpoints/b3/best.pth",
        device=device
    )

    l1_fn = nn.L1Loss()

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    for param in lpips_fn.parameters():
        param.requires_grad = False

    clip_model, _, _ = open_clip.create_model_and_transforms(
        config["clip"]["model_name"],
        pretrained=config["clip"]["pretrained"]
    )

    clip_model = clip_model.to(device)
    clip_model.eval()

    for param in clip_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    resume_path = os.path.join(config["saving"]["checkpoint_dir"], "last.pth")
    best_path = os.path.join(config["saving"]["checkpoint_dir"], "best.pth")

    start_epoch = 1
    best_val_loss = float("inf")

    if os.path.exists(best_path):
        best_checkpoint = torch.load(best_path, map_location=device)
        best_val_loss = best_checkpoint["val_total_loss"]
        print(f"Best previous val loss: {best_val_loss:.6f}")

    try:
        if os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=device)

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            start_epoch = checkpoint["epoch"] + 1

            print(f"Resumed B4 from last checkpoint epoch {checkpoint['epoch']}")

        elif os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=device)

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            start_epoch = checkpoint["epoch"] + 1

            print(f"Resumed B4 from best checkpoint epoch {checkpoint['epoch']}")

        else:
            print("No B4 checkpoint found. Starting from initialized B3 weights.")

    except RuntimeError:
        print("Corrupted last checkpoint detected. Falling back to best checkpoint.")

        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=device)

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            start_epoch = checkpoint["epoch"] + 1

            print(f"Resumed B4 from best checkpoint epoch {checkpoint['epoch']}")
        else:
            print("No valid B4 best checkpoint found. Starting from initialized B3 weights.")

    train_log_path = os.path.join(config["saving"]["log_dir"], "train_log.csv")

    if not os.path.exists(train_log_path) or start_epoch == 1:
        with open(train_log_path, "w") as f:
            f.write(
                "epoch,train_total_loss,train_l1,train_lpips,train_clip,"
                "val_total_loss,val_l1,val_lpips,val_clip\n"
            )
    else:
        print("Existing log file found. Appending resumed training logs.")

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        model.train()

        # keep frozen CLIP encoder in eval mode
        model.clip_encoder.eval()

        total_train_loss = 0.0
        total_train_l1 = 0.0
        total_train_lpips = 0.0
        total_train_clip = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config['training']['epochs']}"
        )

        for batch in progress_bar:
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)

            optimizer.zero_grad()

            pred = model(hazy)

            loss, l1_value, lpips_value, clip_value = compute_loss(
                pred,
                clear,
                l1_fn,
                lpips_fn,
                clip_model,
                config
            )

            if torch.isnan(loss):
                print("NaN loss detected. Stopping training.")
                return

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_train_loss += loss.item()
            total_train_l1 += l1_value
            total_train_lpips += lpips_value
            total_train_clip += clip_value

            progress_bar.set_postfix({
                "loss": loss.item(),
                "l1": l1_value,
                "lpips": lpips_value,
                "clip": clip_value
            })

        n_train = len(train_loader)

        avg_train_loss = total_train_loss / n_train
        avg_train_l1 = total_train_l1 / n_train
        avg_train_lpips = total_train_lpips / n_train
        avg_train_clip = total_train_clip / n_train

        avg_val_loss, avg_val_l1, avg_val_lpips, avg_val_clip = validate(
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
            f"Train Total: {avg_train_loss:.6f} | "
            f"Train L1: {avg_train_l1:.6f} | "
            f"Train LPIPS: {avg_train_lpips:.6f} | "
            f"Train CLIP: {avg_train_clip:.6f} | "
            f"Val Total: {avg_val_loss:.6f} | "
            f"Val L1: {avg_val_l1:.6f} | "
            f"Val LPIPS: {avg_val_lpips:.6f} | "
            f"Val CLIP: {avg_val_clip:.6f}"
        )

        with open(train_log_path, "a") as f:
            f.write(
                f"{epoch},{avg_train_loss},{avg_train_l1},{avg_train_lpips},{avg_train_clip},"
                f"{avg_val_loss},{avg_val_l1},{avg_val_lpips},{avg_val_clip}\n"
            )

        save_samples(
            model=model,
            val_loader=val_loader,
            device=device,
            result_dir=config["saving"]["result_dir"],
            epoch=epoch
        )

        last_path = os.path.join(config["saving"]["checkpoint_dir"], "last.pth")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_total_loss": avg_train_loss,
                "train_l1": avg_train_l1,
                "train_lpips": avg_train_lpips,
                "train_clip": avg_train_clip,
                "val_total_loss": avg_val_loss,
                "val_l1": avg_val_l1,
                "val_lpips": avg_val_lpips,
                "val_clip": avg_val_clip,
            },
            last_path
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            best_path = os.path.join(config["saving"]["checkpoint_dir"], "best.pth")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_total_loss": avg_train_loss,
                    "train_l1": avg_train_l1,
                    "train_lpips": avg_train_lpips,
                    "train_clip": avg_train_clip,
                    "val_total_loss": avg_val_loss,
                    "val_l1": avg_val_l1,
                    "val_lpips": avg_val_lpips,
                    "val_clip": avg_val_clip,
                },
                best_path
            )

            print(
                f"Best B4 model saved at epoch {epoch} "
                f"with val total loss {best_val_loss:.6f}"
            )


if __name__ == "__main__":
    train().n        