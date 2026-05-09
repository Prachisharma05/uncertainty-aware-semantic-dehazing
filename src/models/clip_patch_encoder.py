import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class CLIPPatchEncoder(nn.Module):
    def __init__(
        self,
        model_name="ViT-B-32",
        pretrained="openai",
        input_size=224
    ):
        super().__init__()

        self.input_size = input_size

        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )

        self.visual = self.clip_model.visual

        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.clip_model.eval()

        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

    def preprocess(self, x):
        x = F.interpolate(
            x,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False
        )

        x = (x - self.mean) / self.std
        return x

    def forward(self, x):
        """
        Input:
            x: [B, 3, H, W], range [0,1]

        Output:
            patch_tokens: [B, N, D]
        """

        with torch.no_grad():
            x = self.preprocess(x)

            visual = self.visual

            # Patch embedding
            x = visual.conv1(x)  # [B, width, grid, grid]

            B, C, H, W = x.shape

            x = x.reshape(B, C, H * W)
            x = x.permute(0, 2, 1)  # [B, N, C]

            # Add CLS token
            cls_token = visual.class_embedding.to(x.dtype)
            cls_token = cls_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)

            x = torch.cat([cls_token, x], dim=1)  # [B, 1+N, C]

            # Add positional embedding
            x = x + visual.positional_embedding.to(x.dtype)

            x = visual.patch_dropout(x)
            x = visual.ln_pre(x)

            # Transformer expects [sequence, batch, dim]
            x = x.permute(1, 0, 2)
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)

            # Remove CLS token
            patch_tokens = x[:, 1:, :]  # [B, N, D]

        return patch_tokens