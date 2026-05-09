import os
import yaml
import torch
import torch.nn as nn
import lpips
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.datasets.dehaze_dataset import DehazeDataset
from src.models.unet import UNet
from src.utils.seed import set_seed


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_loss(pred, clear, l1_fn, lpips_fn, lambda_l1, lambda_lpips):
    l1 = l1_fn(pred, clear)

    pred_lpips = pred * 2 - 1
    clear_lpips = clear * 2 - 1

    perceptual = lpips_fn(pred_lpips, clear_lpips).mean()

    total = lambda_l1 * l1 + lambda_lpips * perceptual

    return total, l1.item(), perceptual.item()


def validate(model, val_loader, l1_fn, lpips_fn, config, device):
    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_lpips = 0.0

    lambda_l1 = config["loss"]["lambda_l1"]
    lambda_lpips = config["loss"]["lambda_lpips"]

    with torch.no_grad():
        for batch in val_loader:
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)

            pred = model(hazy)

            loss, l1_value, lpips_value = compute_loss(
                pred,
                clear,
                l1_fn,
                lpips_fn,
                lambda_l1,
                lambda_lpips
            )

            total_loss += loss.item()
            total_l1 += l1_value
            total_lpips += lpips_value

    n = len(val_loader)

    return total_loss / n, total_l1 / n, total_lpips / n


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


def train():
    config = load_config("configs/b2_unet_l1_lpips.yaml")

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

    model = UNet(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"]
    ).to(device)

    l1_fn = nn.L1Loss()

    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    for param in lpips_fn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    b1_checkpoint_path = "checkpoints/b1/best.pth"

    best_val_loss = float("inf")
    start_epoch = 1

    if os.path.exists(b1_checkpoint_path):
        checkpoint = torch.load(b1_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Initialized B2 from B1 checkpoint epoch {checkpoint['epoch']}")
        print(f"B1 val L1 loss: {checkpoint['val_loss']:.6f}")
    else:
        print("B1 checkpoint not found. Training B2 from scratch.")

    train_log_path = os.path.join(config["saving"]["log_dir"], "train_log.csv")

    with open(train_log_path, "w") as f:
        f.write("epoch,train_total_loss,train_l1,train_lpips,val_total_loss,val_l1,val_lpips\n")

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        model.train()

        total_train_loss = 0.0
        total_train_l1 = 0.0
        total_train_lpips = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config['training']['epochs']}"
        )

        for batch in progress_bar:
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)

            optimizer.zero_grad()

            pred = model(hazy)

            loss, l1_value, lpips_value = compute_loss(
                pred,
                clear,
                l1_fn,
                lpips_fn,
                config["loss"]["lambda_l1"],
                config["loss"]["lambda_lpips"]
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

            progress_bar.set_postfix({
                "loss": loss.item(),
                "l1": l1_value,
                "lpips": lpips_value
            })

        n_train = len(train_loader)

        avg_train_loss = total_train_loss / n_train
        avg_train_l1 = total_train_l1 / n_train
        avg_train_lpips = total_train_lpips / n_train

        avg_val_loss, avg_val_l1, avg_val_lpips = validate(
            model,
            val_loader,
            l1_fn,
            lpips_fn,
            config,
            device
        )

        print(
            f"Epoch [{epoch}/{config['training']['epochs']}] "
            f"Train Total: {avg_train_loss:.6f} | "
            f"Train L1: {avg_train_l1:.6f} | "
            f"Train LPIPS: {avg_train_lpips:.6f} | "
            f"Val Total: {avg_val_loss:.6f} | "
            f"Val L1: {avg_val_l1:.6f} | "
            f"Val LPIPS: {avg_val_lpips:.6f}"
        )

        with open(train_log_path, "a") as f:
            f.write(
                f"{epoch},{avg_train_loss},{avg_train_l1},{avg_train_lpips},"
                f"{avg_val_loss},{avg_val_l1},{avg_val_lpips}\n"
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
                "val_total_loss": avg_val_loss,
                "val_l1": avg_val_l1,
                "val_lpips": avg_val_lpips,
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
                    "val_total_loss": avg_val_loss,
                    "val_l1": avg_val_l1,
                    "val_lpips": avg_val_lpips,
                },
                best_path
            )

            print(
                f"Best B2 model saved at epoch {epoch} "
                f"with val total loss {best_val_loss:.6f}"
            )


if __name__ == "__main__":
    train()