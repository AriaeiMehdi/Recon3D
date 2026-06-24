"""
Interactive viewer: raw SPECT volume + PCA patch features side by side.
PCA is upsampled to the ORIGINAL volume size (32x64x64), not just the crop size.
"""
import glob
import os
import sys
import random

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from eval_3d_dino_features import build_model_3d, load_volume


@torch.no_grad()
def extract_features(model, volume, device):
    model.eval()
    batch = volume.unsqueeze(0).to(device)
    out = model(batch, is_training=True)
    return out["x_norm_clstoken"].cpu(), out["x_norm_patchtokens"].cpu()


def pca_features(patch_feats, n_components=3):
    from sklearn.decomposition import PCA
    feats = patch_feats.numpy()
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(feats)
    proj = (proj - proj.min(0)) / (proj.max(0) - proj.min(0) + 1e-8)
    return proj


def get_slice(vol, axis, idx):
    idx = int(idx)
    if axis == 0:
        return vol[idx]
    elif axis == 1:
        return vol[:, idx, :]
    else:
        return vol[:, :, idx]


def max_idx(vol, axis):
    return int(vol.shape[0 if axis == 0 else (1 if axis == 1 else 2)]) - 1


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--data-dir", default="D:/Dr.EntezarMahdi/3D-Recon/data_split")
    ap.add_argument("--out-dir", default="./eval_out")
    ap.add_argument("--pca-cmap", default="hot")
    ap.add_argument("--raw-cmap", default="hot")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = OmegaConf.load(args.config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model_3d(cfg).to(device)
    ckpt_path = max(glob.glob("./output_3d/checkpoint_*.pt"), key=lambda f: int(f.split('_')[-1].split('.')[0]))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["teacher"])
    model.eval()
    print(f"Loaded teacher from {ckpt_path} (iter {ckpt.get('iteration')})")

    # Load ORIGINAL volume (no cropping)
    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.npy")))
    vol_path = random.choice(paths)
    original_vol = load_volume(vol_path)  # [1, D_orig, H_orig, W_orig]
    print(f"Volume: {os.path.basename(vol_path)}, shape: {original_vol.shape}")

    # Crop/pad to model's target size for feature extraction
    target_size = tuple(cfg.crops.global_crops_size)
    model_vol = original_vol.clone()
    # Center crop or pad to target size
    C, D, H, W = model_vol.shape
    tD, tH, tW = target_size
    if D != tD or H != tH or W != tW:
        out = torch.zeros((C, tD, tH, tW), dtype=model_vol.dtype)
        sd, sh, sw = min(D, tD), min(H, tH), min(W, tW)
        td_start = (D - sd) // 2
        th_start = (H - sh) // 2
        tw_start = (W - sw) // 2
        od_start = (tD - sd) // 2
        oh_start = (tH - sh) // 2
        ow_start = (tW - sw) // 2
        out[:, od_start:od_start+sd, oh_start:oh_start+sh, ow_start:ow_start+sw] = \
            model_vol[:, td_start:td_start+sd, th_start:th_start+sh, tw_start:tw_start+sw]
        model_vol = out

    # Extract features from cropped model input
    print("Extracting features...")
    cls_feats, patch_feats = extract_features(model, model_vol, device)
    print(f"CLS: {cls_feats.shape}, Patches: {patch_feats.shape}")

    # PCA on patch grid
    patch_size = cfg.student.patch_size
    grid_shape = tuple(s // patch_size for s in target_size)
    pca_feats = pca_features(patch_feats[0])
    gd, gh, gw = grid_shape
    pca_vol = pca_feats.reshape(gd, gh, gw, 3)
    pca_gray = pca_vol[:, :, :, 0]

    # Upsample PCA to global crop size
    raw_vol = model_vol[0].numpy()  # [target_D, target_H, target_W]
    pca_tensor = torch.from_numpy(pca_gray).float().unsqueeze(0).unsqueeze(0)
    pca_upscaled = F.interpolate(
        pca_tensor, size=raw_vol.shape, mode="trilinear", align_corners=False
    ).squeeze().numpy()
    pca_upscaled = np.clip(pca_upscaled, 0, 1)

    print(f"PCA grid: {pca_gray.shape} -> upsampled to original size: {pca_upscaled.shape}")

    # Interactive viewer
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, RadioButtons

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(bottom=0.25)

    mid = raw_vol.shape[0] // 2

    im_pca = axes[0].imshow(pca_upscaled[mid], cmap=args.pca_cmap)
    axes[0].set_title(f'PCA - Axial {mid}/{raw_vol.shape[0]-1}')
    axes[0].axis('off')

    im_raw = axes[1].imshow(raw_vol[mid], cmap=args.raw_cmap)
    axes[1].set_title(f'Raw {tuple(raw_vol.shape)} - Axial {mid}/{raw_vol.shape[0]-1}')
    axes[1].axis('off')

    ax_slider = plt.axes([0.15, 0.10, 0.7, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, raw_vol.shape[0] - 1, valinit=mid, valstep=1)

    ax_radio = plt.axes([0.88, 0.3, 0.11, 0.25])
    radio = RadioButtons(ax_radio, ['Axial', 'Coronal', 'Sagittal'], active=0)

    axis_names = ['Axial', 'Coronal', 'Sagittal']

    def update(val):
        idx = int(slider.val)
        axis_idx = axis_names.index(radio.value_selected)
        axis_name = radio.value_selected

        raw_slc = get_slice(raw_vol, axis_idx, idx)
        pca_slc = get_slice(pca_upscaled, axis_idx, idx)

        im_raw.set_data(raw_slc)
        im_pca.set_data(pca_slc)

        raw_max = max_idx(raw_vol, axis_idx)
        axes[0].set_title(f'PCA - {axis_name} {idx}/{raw_max}')
        axes[1].set_title(f'Raw {tuple(raw_vol.shape)} - {axis_name} {idx}/{raw_max}')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    radio.on_clicked(update)
    plt.show()


if __name__ == "__main__":
    main()
