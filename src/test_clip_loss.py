import torch
import torch.nn.functional as F
import open_clip


device = "cuda" if torch.cuda.is_available() else "cpu"


def clip_normalize(img):
    mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073],
        device=img.device
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711],
        device=img.device
    ).view(1, 3, 1, 1)

    return (img - mean) / std


model, _, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

model = model.to(device)
model.eval()

for param in model.parameters():
    param.requires_grad = False


img1 = torch.rand(2, 3, 256, 256).to(device)
img2 = torch.rand(2, 3, 256, 256).to(device)

img1 = F.interpolate(img1, size=(224, 224), mode="bilinear", align_corners=False)
img2 = F.interpolate(img2, size=(224, 224), mode="bilinear", align_corners=False)

img1 = clip_normalize(img1)
img2 = clip_normalize(img2)

with torch.no_grad():
    feat1 = model.encode_image(img1)
    feat2 = model.encode_image(img2)

feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)

clip_loss = 1 - F.cosine_similarity(feat1, feat2, dim=-1).mean()

print("Feature 1 shape:", feat1.shape)
print("Feature 2 shape:", feat2.shape)
print("CLIP loss:", clip_loss.item())