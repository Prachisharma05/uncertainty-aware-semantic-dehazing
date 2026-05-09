import torch
import lpips


device = "cuda" if torch.cuda.is_available() else "cpu"

lpips_fn = lpips.LPIPS(net="alex").to(device)
lpips_fn.eval()

img1 = torch.rand(2, 3, 256, 256).to(device)
img2 = torch.rand(2, 3, 256, 256).to(device)

# LPIPS expects images in [-1, 1]
img1_lpips = img1 * 2 - 1
img2_lpips = img2 * 2 - 1

with torch.no_grad():
    loss = lpips_fn(img1_lpips, img2_lpips).mean()

print("LPIPS loss:", loss.item())