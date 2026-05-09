from torch.utils.data import DataLoader
from src.datasets.dehaze_dataset import DehazeDataset


train_dataset = DehazeDataset(
    pairs_file="data/RESIDE/ITS/train_pairs.txt",
    image_size=256
)

val_dataset = DehazeDataset(
    pairs_file="data/RESIDE/ITS/val_pairs.txt",
    image_size=256
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

train_batch = next(iter(train_loader))
val_batch = next(iter(val_loader))

print("Train dataset size:", len(train_dataset))
print("Val dataset size:", len(val_dataset))

print("Train hazy shape:", train_batch["hazy"].shape)
print("Train clear shape:", train_batch["clear"].shape)

print("Val hazy shape:", val_batch["hazy"].shape)
print("Val clear shape:", val_batch["clear"].shape)

print("Train hazy min/max:", train_batch["hazy"].min().item(), train_batch["hazy"].max().item())
print("Train clear min/max:", train_batch["clear"].min().item(), train_batch["clear"].max().item())