import os
import pandas as pd


FILES = {
    "B1_UNet_L1": "results/b1/b1_metrics.csv",
    "B2_UNet_L1_LPIPS": "results/b2/b2_metrics.csv",
    "B3_UNet_L1_LPIPS_CLIP": "results/b3/b3_metrics.csv",
    "B4_UNet_CrossAttention": "results/b4/b4_metrics.csv",
    "B5_GatedCrossAttention": "results/b5/b5_metrics.csv",
    "B6_FrozenBackbone_GatedCrossAttention": "results/b6/b6_metrics.csv",
    "B6_Ablation_NoSemantics": "results/b6/b6_ablation_metrics.csv",
}


def main():
    rows = []

    for model_name, path in FILES.items():
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue

        df = pd.read_csv(path)
        row = df.iloc[0].to_dict()

        rows.append({
            "Model": model_name,
            "Checkpoint Epoch": row.get("checkpoint_epoch", None),
            "PSNR ↑": row.get("psnr", None),
            "SSIM ↑": row.get("ssim", None),
            "LPIPS ↓": row.get("lpips", None),
            "Val L1 ↓": row.get("val_l1_loss", None),
            "Val CLIP ↓": row.get("val_clip_loss", None),
            "Gate Mean": row.get("val_gate_mean", None),
        })

    benchmark_rows = [
        {"Model": "FFA-Net (SOTA)", "PSNR ↑": 36.39, "SSIM ↑": 0.9888},
        {"Model": "AECR-Net (SOTA)", "PSNR ↑": 37.17, "SSIM ↑": 0.9901},
        {"Model": "DehazeFormer-S (SOTA)", "PSNR ↑": 37.84, "SSIM ↑": 0.9912},
    ]
    rows.extend(benchmark_rows)

    result_df = pd.DataFrame(rows)

    os.makedirs("results/final", exist_ok=True)

    result_df.to_csv("results/final/final_model_comparison.csv", index=False)

    print("\nFinal Model Comparison:")
    print(result_df)


if __name__ == "__main__":
    main()