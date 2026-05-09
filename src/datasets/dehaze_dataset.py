from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class DehazeDataset(Dataset):
    def __init__(self, pairs_file, image_size=256):
        self.pairs = []

        with open(pairs_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            hazy_path, clear_path = line.strip().split(",")
            self.pairs.append((hazy_path, clear_path))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hazy_path, clear_path = self.pairs[idx]

        hazy_img = Image.open(hazy_path).convert("RGB")
        clear_img = Image.open(clear_path).convert("RGB")

        hazy_tensor = self.transform(hazy_img)
        clear_tensor = self.transform(clear_img)

        return {
            "hazy": hazy_tensor,
            "clear": clear_tensor,
            "hazy_path": hazy_path,
            "clear_path": clear_path
        }