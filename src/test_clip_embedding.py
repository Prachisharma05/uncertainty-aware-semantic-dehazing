import torch
import open_clip


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

model = model.to(device)
model.eval()

# Create dummy image
img = torch.rand(1, 3, 224, 224).to(device)

# Normalize like CLIP expects
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1).to(device)
std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1).to(device)

img = (img - mean) / std

# Forward pass
with torch.no_grad():
    features = model.encode_image(img)

# Normalize embedding
features = features / features.norm(dim=-1, keepdim=True)

print("Embedding shape:", features.shape)
print("Embedding norm:", features.norm(dim=-1))