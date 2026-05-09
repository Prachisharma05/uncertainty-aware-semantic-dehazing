import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        unet_channels=1024,
        clip_dim=768,
        attn_dim=512,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.unet_channels = unet_channels

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

        self.norm = nn.LayerNorm(unet_channels)

    def forward(self, unet_feat, clip_tokens):
        """
        unet_feat:
            [B, C, H, W]

        clip_tokens:
            [B, N, D]

        output:
            [B, C, H, W]
        """

        B, C, H, W = unet_feat.shape

        # [B, C, H, W] -> [B, H*W, C]
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

        # Residual + normalization
        fused_tokens = self.norm(unet_tokens + attn_out)

        # [B, H*W, C] -> [B, C, H, W]
        fused_feat = fused_tokens.permute(0, 2, 1).reshape(B, C, H, W)

        return fused_feat, attn_weights