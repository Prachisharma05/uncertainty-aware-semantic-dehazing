import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.datasets.dehaze_dataset import DehazeDataset
from src.models.unet import UNet
from src.utils.seed import set_seed


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)

            pred = model(hazy)
            loss = criterion(pred, clear)

            total_loss += loss.item()

    return total_loss / len(val_loader)


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
    config = load_config("configs/b1_unet_l1.yaml")

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

    criterion = nn.L1Loss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=config["training"]["mixed_precision"] and device == "cuda"
    )

    resume_path = os.path.join(config["saving"]["checkpoint_dir"], "best.pth")

    start_epoch = 1
    best_val_loss = float("inf")

    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]

        print(f"Resumed from epoch {checkpoint['epoch']}")
        print(f"Best val loss so far: {best_val_loss:.6f}")

    train_log_path = os.path.join(config["saving"]["log_dir"], "train_log.csv")

    with open(train_log_path, "w") as f:
        f.write("epoch,train_loss,val_loss\n")

    for epoch in range(start_epoch, config["training"]["epochs"] + 1):
        model.train()
        total_train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']}")

        for batch in progress_bar:
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(
                enabled=config["training"]["mixed_precision"] and device == "cuda"
            ):
                pred = model(hazy)
                loss = criterion(pred, clear)
                if torch.isnan(loss):
                    print("NaN loss detected. Stopping training.")
                    return

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch}/{config['training']['epochs']}] "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f}"
        )

        with open(train_log_path, "a") as f:
            f.write(f"{epoch},{avg_train_loss},{avg_val_loss}\n")

        save_samples(
            model=model,
            val_loader=val_loader,
            device=device,
            result_dir=config["saving"]["result_dir"],
            epoch=epoch
        )

        last_path = os.path.join(
            config["saving"]["checkpoint_dir"],
            "last.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            },
            last_path
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            best_path = os.path.join(
                config["saving"]["checkpoint_dir"],
                "best.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                best_path
            )

            print(f"Best model saved at epoch {epoch} with val loss {best_val_loss:.6f}")


if __name__ == "__main__":
    train()