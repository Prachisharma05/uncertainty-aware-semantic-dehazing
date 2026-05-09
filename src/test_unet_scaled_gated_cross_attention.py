import torch

from src.models.unet_scaled_gated_cross_attention import UNetScaledGatedCrossAttention


device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNetScaledGatedCrossAttention(
    in_channels=3,
    out_channels=3,
    clip_model_name="ViT-B-16",
    clip_pretrained="openai",
    clip_input_size=224,
    alpha=0.1
).to(device)

model.eval()

x = torch.rand(1, 3, 256, 256).to(device)

with torch.no_grad():
    y, attn, gate = model(
        x,
        return_attention=True,
        return_gate=True
    )

print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Attention shape:", attn.shape)
print("Gate shape:", gate.shape)
print("Output min/max:", y.min().item(), y.max().item())
print("Gate min/max:", gate.min().item(), gate.max().item())
print("Gate mean:", gate.mean().item())