# Uncertainty-Aware Semantic-Guided Image Dehazing

This project implements a semantic-guided image dehazing framework using UNet, perceptual loss, CLIP semantic supervision, patch-level CLIP ViT tokens, cross-attention, and uncertainty-aware gating.

## Current Progress

### Phase 0: Data Pipeline
- RESIDE ITS dataset organized
- Pairing validation completed
- Train/validation split created
- DataLoader tested
- Visual pairing verified

### Phase 1: B1 - UNet + L1
- Baseline UNet trained
- Best PSNR: 33.8155
- SSIM: 0.9706
- LPIPS: 0.01494

### Phase 2: B2 - UNet + L1 + LPIPS
- Perceptual loss added
- PSNR: 36.0380
- SSIM: 0.9753
- LPIPS: 0.01067

### Phase 3: B3 - UNet + L1 + LPIPS + CLIP Loss
- Global CLIP semantic loss added
- PSNR: 36.2839
- SSIM: 0.9758
- LPIPS: 0.01031

### Phase 4: CLIP Patch Token Extraction
- CLIP ViT-B/16 patch tokens extracted
- Patch token shape: [B, 196, 768]

### Phase 5: B4 - Cross-Attention
- Cross-attention module implemented
- Spatial semantic injection tested

### Phase 6: B5 & B6 - Gated Cross-Attention
- Uncertainty-aware gated semantic fusion completed.
- B6 frozen backbone variations successfully trained.

### Phase 7: Final Evaluations & Benchmarking
- **Semantic Token Ablation**: Validated that removing CLIP semantic tokens leads to significant performance drop.
- **Explainability**: Implemented alpha-blended heatmap overlays showing precisely where the semantic gate activates.
- **CLIP Semantic Similarity**: Validated semantic preservation. B6 achieves `0.9870` cosine similarity compared to baseline B1 (`0.9787`).
- **SOTA Comparison**: Evaluated against FFA-Net, AECR-Net, and DehazeFormer-S using RESIDE ITS.

## Tech Stack

- Python
- PyTorch
- OpenCLIP
- LPIPS
- torchvision
- scikit-image
- OpenCV
- Matplotlib
- YAML configuration

## Project Structure

```text
src/
├── datasets/
├── models/
├── utils/
├── train_b1.py
├── train_b2.py
├── train_b3.py
├── train_b4.py
├── train_b5.py
├── evaluate_b1.py
├── evaluate_b2.py
├── evaluate_b3.py
├── evaluate_b4.py
├── evaluate_b5.py
├── evaluate_b6.py
├── evaluate_b6_ablation.py
├── evaluate_clip_similarity.py
└── visualize_b6_gate.py