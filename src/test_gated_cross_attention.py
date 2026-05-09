import torch

from src.models.gated_cross_attention import GatedCrossAttentionBlock


device = "cuda" if torch.cuda.is_available() else "cpu"

block = GatedCrossAttentionBlock(
    unet_channels=1024,
    clip_dim=768,
    attn_dim=512,
    num_heads=8
).to(device)

unet_feat = torch.randn(2, 1024, 16, 16).to(device)
clip_tokens = torch.randn(2, 196, 768).to(device)

fused_feat, attn_weights, gate_map = block(
    unet_feat,
    clip_tokens,
    return_gate=True
)

print("UNet feature shape:", unet_feat.shape)
print("CLIP token shape:", clip_tokens.shape)
print("Fused feature shape:", fused_feat.shape)
print("Attention shape:", attn_weights.shape)
print("Gate map shape:", gate_map.shape)
print("Gate min/max:", gate_map.min().item(), gate_map.max().item())
print("Gate mean:", gate_map.mean().item())