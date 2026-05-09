import torch

from src.models.unet_cross_attention import UNetCrossAttention


device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNetCrossAttention(
    in_channels=3,
    out_channels=3,
    clip_model_name="ViT-B-16",
    clip_pretrained="openai",
    clip_input_size=224
).to(device)

model.eval()

x = torch.rand(1, 3, 256, 256).to(device)

with torch.no_grad():
    y, attn = model(x, return_attention=True)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Attention shape:", attn.shape)
print("Output min/max:", y.min().item(), y.max().item())