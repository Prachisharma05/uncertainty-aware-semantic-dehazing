import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ultralytics import YOLO


from src.datasets.dehaze_dataset import DehazeDataset
from src.models.unet import UNet
from src.models.unet_gated_cross_attention import UNetGatedCrossAttention


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def tensor_to_uint8_image(tensor):
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = (img.clip(0, 1) * 255).astype("uint8")
    img = np.ascontiguousarray(img)
    return img

def run_yolo(yolo, img, conf=0.25):
    results = yolo.predict(img, conf=conf, verbose=False)
    result = results[0]

    boxes = result.boxes
    count = 0 if boxes is None else len(boxes)

    avg_conf = 0.0
    if boxes is not None and len(boxes) > 0:
        avg_conf = float(boxes.conf.mean().item())

    annotated = result.plot()
    return annotated, count, avg_conf


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("results/phase8_detection", exist_ok=True)

    b3_config = load_config("configs/b3_unet_clip_loss.yaml")
    b6_config = load_config("configs/b6_frozen_backbone.yaml")

    dataset = DehazeDataset(
        pairs_file=b6_config["data"]["val_pairs"],
        image_size=b6_config["data"]["image_size"]
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    b3 = UNet(
        in_channels=b3_config["model"]["in_channels"],
        out_channels=b3_config["model"]["out_channels"]
    ).to(device)

    b3_ckpt = torch.load("checkpoints/b3/best.pth", map_location=device)
    b3.load_state_dict(b3_ckpt["model_state_dict"])
    b3.eval()

    b6 = UNetGatedCrossAttention(
        in_channels=b6_config["model"]["in_channels"],
        out_channels=b6_config["model"]["out_channels"],
        clip_model_name=b6_config["clip"]["model_name"],
        clip_pretrained=b6_config["clip"]["pretrained"],
        clip_input_size=b6_config["clip"]["input_size"]
    ).to(device)

    b6_ckpt = torch.load("checkpoints/b6/best.pth", map_location=device)
    b6.load_state_dict(b6_ckpt["model_state_dict"])
    b6.eval()
    b6.clip_encoder.eval()

    yolo = YOLO("yolov8n.pt")

    summary_rows = []

    max_images = 10

    for idx, batch in enumerate(loader):
        if idx >= max_images:
            break

        hazy = batch["hazy"].to(device)
        clear = batch["clear"].to(device)

        with torch.no_grad():
            b3_out = b3(hazy)
            b6_out = b6(hazy)

        images = {
            "Hazy": tensor_to_uint8_image(hazy[0]),
            "B3": tensor_to_uint8_image(b3_out[0]),
            "B6": tensor_to_uint8_image(b6_out[0]),
            "GT": tensor_to_uint8_image(clear[0]),
        }

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        row = {"image_id": idx}

        for ax, (name, img) in zip(axes, images.items()):
            annotated, count, avg_conf = run_yolo(yolo, img, conf=0.25)

            ax.imshow(annotated)
            ax.set_title(f"{name}\nObjects: {count}, Conf: {avg_conf:.3f}")
            ax.axis("off")

            row[f"{name}_count"] = count
            row[f"{name}_avg_conf"] = avg_conf

        summary_rows.append(row)

        plt.tight_layout()
        save_path = f"results/phase8_detection/detection_compare_{idx}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved: {save_path}")

    import pandas as pd

    df = pd.DataFrame(summary_rows)
    df.to_csv("results/phase8_detection/detection_summary.csv", index=False)

    print("\nDetection Summary:")
    print(df)


if __name__ == "__main__":
    main()