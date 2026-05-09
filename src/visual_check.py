import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.datasets.dehaze_dataset import DehazeDataset
from src.utils.seed import set_seed


def show_batch():
    set_seed(42)

    dataset = DehazeDataset(
        pairs_file="data/RESIDE/ITS/val_pairs.txt",
        image_size=256
    )

    loader = DataLoader(dataset, batch_size=6, shuffle=False)

    batch = next(iter(loader))

    hazy = batch["hazy"]
    clear = batch["clear"]

    plt.figure(figsize=(8, 14))

    for i in range(6):
        hazy_img = hazy[i].permute(1, 2, 0)
        clear_img = clear[i].permute(1, 2, 0)

        plt.subplot(6, 2, 2 * i + 1)
        plt.imshow(hazy_img)
        plt.title("Hazy")
        plt.axis("off")

        plt.subplot(6, 2, 2 * i + 2)
        plt.imshow(clear_img)
        plt.title("Clean")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("results/phase0/final_pair_visual_check.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    show_batch()