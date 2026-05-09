import torch
import torch.nn as nn


class ScaledGatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        unet_channels=1024,
        clip_dim=768,
        attn_dim=512,
        num_heads=8,
        dropout=0.1,
        alpha=0.1
    ):
        super().__init__()

        self.unet_channels = unet_channels
        self.alpha = alpha

        self.query_proj = nn.Linear(unet_channels, attn_dim)
        self.key_proj = nn.Linear(clip_dim, attn_dim)
        self.value_proj = nn.Linear(clip_dim, attn_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.out_proj = nn.Linear(attn_dim, unet_channels)

        self.gate = nn.Sequential(
            nn.Conv2d(unet_channels * 2, unet_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(unet_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(unet_channels)

    def forward(self, unet_feat, clip_tokens, return_gate=False):
        B, C, H, W = unet_feat.shape

        unet_tokens = unet_feat.flatten(2).permute(0, 2, 1)

        q = self.query_proj(unet_tokens)
        k = self.key_proj(clip_tokens)
        v = self.value_proj(clip_tokens)

        attn_out, attn_weights = self.attn(
            query=q,
            key=k,
            value=v,
            need_weights=True
        )

        attn_out = self.out_proj(attn_out)
        attn_feat = attn_out.permute(0, 2, 1).reshape(B, C, H, W)

        gate_input = torch.cat([unet_feat, attn_feat], dim=1)
        gate_map = self.gate(gate_input)

        # Key B7 change: scaled semantic residual
        fused_feat = unet_feat + self.alpha * gate_map * attn_feat

        fused_tokens = fused_feat.flatten(2).permute(0, 2, 1)
        fused_tokens = self.norm(fused_tokens)
        fused_feat = fused_tokens.permute(0, 2, 1).reshape(B, C, H, W)

        if return_gate:
            return fused_feat, attn_weights, gate_map

        return fused_feat, attn_weights