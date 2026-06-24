"""Visualize augmentations on a real SPECT volume using actual training parameters."""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from dinov3.data.augmentations_3d import DataAugmentation3D_DINO

# Load training config
cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), "..", "dinov3", "configs", "train", "dinov3_3d_spect_vits16.yaml"))

# Load a random volume
data_dir = r"D:\Dr.EntezarMahdi\3D-Recon\data_split_cropped"
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
vol_path = os.path.join(data_dir, random.choice(files))
vol = torch.from_numpy(np.load(vol_path)).unsqueeze(0).float()
print(f"Volume: {os.path.basename(vol_path)}, shape: {vol.shape}")

# Build augmentation pipeline with ACTUAL training parameters
aug = DataAugmentation3D_DINO(
    global_crops_scale=cfg.crops.global_crops_scale,
    local_crops_scale=cfg.crops.local_crops_scale,
    local_crops_number=8,
    global_crops_size=tuple(cfg.crops.global_crops_size),
    local_crops_size=tuple(cfg.crops.local_crops_size),
    rotation_angle_range=tuple(cfg.crops.rotation_angle_range),
    rotation_p=cfg.crops.rotation_p,
)

# Apply augmentation
output = aug(vol)

# Plot: original + 2 global crops + 8 local crops
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# Row 0: Original + 2 global crops
axes[0, 0].imshow(vol[0, vol.shape[1]//2].numpy(), cmap="hot")
axes[0, 0].set_title(f"Original\n{tuple(vol.shape)}")
axes[0, 0].axis("off")

for i in range(2):
    g = output["global_crops"][i]
    axes[0, i+1].imshow(g[0, g.shape[1]//2].cpu().numpy(), cmap="hot")
    axes[0, i+1].set_title(f"Global Crop {i+1}\n{tuple(g.shape)}")
    axes[0, i+1].axis("off")
axes[0, 3].axis("off")

# Row 1: Local crops 1-4
for i in range(4):
    lc = output["local_crops"][i]
    axes[1, i].imshow(lc[0, lc.shape[1]//2].cpu().numpy(), cmap="hot")
    axes[1, i].set_title(f"Local Crop {i+1}\n{tuple(lc.shape)}")
    axes[1, i].axis("off")

# Row 2: Local crops 5-8
for i in range(4):
    lc = output["local_crops"][i+4]
    axes[2, i].imshow(lc[0, lc.shape[1]//2].cpu().numpy(), cmap="hot")
    axes[2, i].set_title(f"Local Crop {i+5}\n{tuple(lc.shape)}")
    axes[2, i].axis("off")

plt.suptitle("DINOv3 Augmentations (Training Config): 2 Global + 8 Local Crops", fontsize=14)
plt.tight_layout()
plt.savefig(r"D:\Dr.EntezarMahdi\3D-Recon\augment_test.png", dpi=150)
print(f"Saved: augment_test.png")
