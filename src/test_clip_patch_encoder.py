import torch
from src.models.clip_patch_encoder import CLIPPatchEncoder


device = "cuda" if torch.cuda.is_available() else "cpu"

encoder = CLIPPatchEncoder(
    model_name="ViT-B-16",
    pretrained="openai",
    input_size=224
).to(device)

encoder.eval()

x = torch.rand(2, 3, 256, 256).to(device)

tokens = encoder(x)

print("Input shape:", x.shape)
print("Patch token shape:", tokens.shape)
print("Requires grad:", tokens.requires_grad)
print("Token min/max:", tokens.min().item(), tokens.max().item())