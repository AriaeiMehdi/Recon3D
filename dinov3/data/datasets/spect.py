import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("dinov3")


class SPECT3D(Dataset):
    """
    Dataset for 3D SPECT volumes in NIfTI format (.nii.gz).

    Expected directory structure:
        root/
            subject_001/scan.nii.gz
            subject_002/scan.nii.gz
            ...

    Or flat structure:
        root/
            scan_001.nii.gz
            scan_002.nii.gz
            ...
    """

    def __init__(
        self,
        *,
        root: str,
        split: str = "TRAIN",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        target_size: Optional[Tuple[int, int, int]] = None,
        normalize: str = "zscore",
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.target_size = target_size
        self.normalize = normalize

        self.samples = self._find_samples(root)
        if len(self.samples) == 0:
            raise RuntimeError(f"No .nii.gz files found in {root}")

        logger.info(f"SPECT3D dataset: found {len(self.samples)} volumes in {root}")

    def _find_samples(self, root: str):
        samples = []
        root_path = Path(root)

        # Try .npy files first (flat structure)
        for f in sorted(root_path.glob("*.npy")):
            samples.append(str(f))

        # Try .nii.gz files (flat structure)
        if not samples:
            for f in sorted(root_path.glob("*.nii.gz")):
                samples.append(str(f))

        # Try subdirectory structure
        if not samples:
            for d in sorted(root_path.iterdir()):
                if d.is_dir():
                    dir_samples = list(d.glob("*.npy"))
                    if not dir_samples:
                        dir_samples = list(d.glob("*.nii.gz"))
                    samples.extend(str(f) for f in sorted(dir_samples))

        return samples

    def _load_volume(self, path: str) -> np.ndarray:
        if path.endswith('.npy'):
            volume = np.load(path).astype(np.float32)
        else:
            try:
                import nibabel as nib
            except ImportError:
                raise ImportError("nibabel is required for .nii.gz files. Install with: pip install nibabel")
            img = nib.load(path)
            volume = img.get_fdata().astype(np.float32)
        return volume

    def _normalize(self, volume: np.ndarray) -> np.ndarray:
        if self.normalize == "zscore":
            mean = volume.mean()
            std = volume.std()
            if std > 0:
                volume = (volume - mean) / std
        elif self.normalize == "minmax":
            vmin = volume.min()
            vmax = volume.max()
            if vmax > vmin:
                volume = (volume - vmin) / (vmax - vmin)
        elif self.normalize == "none":
            pass
        return volume

    def _center_crop_or_pad(self, volume: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        D, H, W = volume.shape
        tD, tH, tW = target_size

        # Center crop
        if D > tD:
            start = (D - tD) // 2
            volume = volume[start : start + tD]
        if H > tH:
            start = (H - tH) // 2
            volume = volume[:, start : start + tH]
        if W > tW:
            start = (W - tW) // 2
            volume = volume[:, :, start : start + tW]

        # Pad with zeros
        D, H, W = volume.shape
        pad_d = max(0, tD - D)
        pad_h = max(0, tH - H)
        pad_w = max(0, tW - W)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            volume = np.pad(
                volume,
                (
                    (pad_d // 2, pad_d - pad_d // 2),
                    (pad_h // 2, pad_h - pad_h // 2),
                    (pad_w // 2, pad_w - pad_w // 2),
                ),
                mode="constant",
                constant_values=0,
            )

        return volume

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        path = self.samples[index]
        volume = self._load_volume(path)
        volume = self._normalize(volume)

        if self.target_size is not None:
            volume = self._center_crop_or_pad(volume, self.target_size)

        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        volume_tensor = torch.from_numpy(volume).unsqueeze(0)

        if self.transform is not None:
            volume_tensor = self.transform(volume_tensor)

        target = ()
        if self.target_transform is not None:
            target = self.target_transform(target)

        return volume_tensor, target

    def __len__(self) -> int:
        return len(self.samples)
