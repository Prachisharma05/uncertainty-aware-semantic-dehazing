import pandas as pd


def main():
    csv_path = "results/phase8_detection/detection_summary.csv"

    df = pd.read_csv(csv_path)

    print("\n--- Detection Summary Analysis ---")

    avg_counts = {
        "Hazy_avg_count": df["Hazy_count"].mean(),
        "B3_avg_count": df["B3_count"].mean(),
        "B6_avg_count": df["B6_count"].mean(),
        "GT_avg_count": df["GT_count"].mean(),
    }

    avg_conf = {
        "Hazy_avg_conf": df["Hazy_avg_conf"].mean(),
        "B3_avg_conf": df["B3_avg_conf"].mean(),
        "B6_avg_conf": df["B6_avg_conf"].mean(),
        "GT_avg_conf": df["GT_avg_conf"].mean(),
    }

    b6_better_conf = (df["B6_avg_conf"] > df["B3_avg_conf"]).sum()
    b3_better_conf = (df["B3_avg_conf"] > df["B6_avg_conf"]).sum()
    equal_conf = (df["B3_avg_conf"] == df["B6_avg_conf"]).sum()

    b6_better_count = (df["B6_count"] > df["B3_count"]).sum()
    b3_better_count = (df["B3_count"] > df["B6_count"]).sum()
    equal_count = (df["B3_count"] == df["B6_count"]).sum()

    total = len(df)

    results = {
        **avg_counts,
        **avg_conf,
        "B6_better_conf_samples": b6_better_conf,
        "B3_better_conf_samples": b3_better_conf,
        "Equal_conf_samples": equal_conf,
        "B6_better_conf_percent": (b6_better_conf / total) * 100,
        "B6_better_count_samples": b6_better_count,
        "B3_better_count_samples": b3_better_count,
        "Equal_count_samples": equal_count,
        "B6_better_count_percent": (b6_better_count / total) * 100,
    }

    result_df = pd.DataFrame([results])

    result_df.to_csv(
        "results/phase8_detection/detection_analysis.csv",
        index=False
    )

    print(result_df.T)


if __name__ == "__main__":
    main()