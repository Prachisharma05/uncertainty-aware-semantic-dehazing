import torch
from src.models.unet import UNet

model = UNet().cuda()

x = torch.randn(1, 3, 256, 256).cuda()

y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Output min/max:", y.min().item(), y.max().item())