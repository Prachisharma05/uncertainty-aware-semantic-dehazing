import os
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "results/final/final_all_results.csv"
OUT_DIR = "results/final/figures"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df.dropna(how="all")


def save_bar_chart(data, x_col, y_col, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.bar(data[x_col], data[y_col])
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=300)
    plt.close()


def save_metric_group_chart(data, dataset_name, filename):
    metrics = ["PSNR", "SSIM", "LPIPS"]

    for metric in metrics:
        plt.figure(figsize=(7, 5))
        plt.bar(data["Model"], data[metric])
        plt.title(f"{dataset_name} {metric} Comparison")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUT_DIR, f"{filename}_{metric.lower()}.png"),
            dpi=300
        )
        plt.close()


# Figure 1: RESIDE ablation PSNR
reside = df[df["Dataset"] == "RESIDE"]
save_bar_chart(
    reside,
    "Model",
    "PSNR",
    "RESIDE Ablation Study: PSNR",
    "PSNR ↑",
    "reside_ablation_psnr.png"
)

# Figure 2: RESIDE LPIPS
save_bar_chart(
    reside,
    "Model",
    "LPIPS",
    "RESIDE Ablation Study: LPIPS",
    "LPIPS ↓",
    "reside_ablation_lpips.png"
)

# Figure 3: NH-HAZE comparison
nh = df[df["Dataset"] == "NH-HAZE"]
save_metric_group_chart(
    nh,
    "NH-HAZE",
    "nh_haze_comparison"
)

# Figure 4: Dense-Haze comparison
dense = df[df["Dataset"] == "DENSE-HAZE"]
save_metric_group_chart(
    dense,
    "Dense-Haze",
    "dense_haze_comparison"
)

# Figure 5: RTTS detection confidence
rtts = df[df["Dataset"] == "RTTS"]
save_bar_chart(
    rtts,
    "Model",
    "DetectionConfidence",
    "RTTS Downstream Detection Confidence",
    "Average YOLO Confidence ↑",
    "rtts_detection_confidence.png"
)

print("Final paper figures saved to:", OUT_DIR)