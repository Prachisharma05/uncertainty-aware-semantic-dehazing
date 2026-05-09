import torch

from src.models.cross_attention import CrossAttentionBlock


device = "cuda" if torch.cuda.is_available() else "cpu"

block = CrossAttentionBlock(
    unet_channels=1024,
    clip_dim=768,
    attn_dim=512,
    num_heads=8
).to(device)

unet_feat = torch.randn(2, 1024, 16, 16).to(device)
clip_tokens = torch.randn(2, 196, 768).to(device)

fused_feat, attn_weights = block(unet_feat, clip_tokens)

print("UNet feature shape:", unet_feat.shape)
print("CLIP token shape:", clip_tokens.shape)
print("Fused feature shape:", fused_feat.shape)
print("Attention weight shape:", attn_weights.shape)
print("Fused min/max:", fused_feat.min().item(), fused_feat.max().item())