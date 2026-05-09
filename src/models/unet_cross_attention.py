import torch
import torch.nn as nn

from src.models.unet import ConvBlock
from src.models.clip_patch_encoder import CLIPPatchEncoder
from src.models.cross_attention import CrossAttentionBlock


class UNetCrossAttention(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        clip_model_name="ViT-B-16",
        clip_pretrained="openai",
        clip_input_size=224
    ):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Frozen CLIP patch encoder
        self.clip_encoder = CLIPPatchEncoder(
            model_name=clip_model_name,
            pretrained=clip_pretrained,
            input_size=clip_input_size
        )

        # Cross-attention semantic fusion
        self.cross_attention = CrossAttentionBlock(
            unet_channels=1024,
            clip_dim=768,
            attn_dim=512,
            num_heads=8,
            dropout=0.1
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, return_attention=False):
        """
        Input:
            x: hazy image [B, 3, 256, 256]

        Output:
            out: dehazed image [B, 3, 256, 256]
        """

        # CLIP patch tokens from hazy image
        clip_tokens = self.clip_encoder(x)  # [B, 196, 768]

        # UNet Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # [B, 1024, 16, 16]

        # Cross-attention injection
        b_fused, attn_weights = self.cross_attention(b, clip_tokens)

        # Decoder
        d4 = self.up4(b_fused)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        out = torch.sigmoid(out)

        if return_attention:
            return out, attn_weights

        return out