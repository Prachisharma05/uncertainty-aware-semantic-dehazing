import torch
import torch.nn as nn

from src.models.unet import UNet


device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)

hazy = torch.randn(2, 3, 256, 256).to(device)
clear = torch.rand(2, 3, 256, 256).to(device)

pred = model(hazy)

criterion = nn.L1Loss()
loss = criterion(pred, clear)

print("Prediction shape:", pred.shape)
print("Target shape:", clear.shape)
print("L1 loss:", loss.item())