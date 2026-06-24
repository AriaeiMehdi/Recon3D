import logging
import math
import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

logger = logging.getLogger("dinov3")


def _make_3tuple(x):
    if hasattr(x, '__iter__') and not isinstance(x, str):
        x = list(x)
        assert len(x) == 3
        return tuple(x)
    return (x, x, x)


class RandomCrop3D:
    """Random crop of a 3D volume based on a fraction of the ORIGINAL volume."""

    def __init__(self, size: Union[int, Tuple[int, int, int]], scale: Tuple[float, float] = (0.5, 1.0)):
        self.size = _make_3tuple(size)
        self.scale = scale

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        C, D, H, W = volume.shape
        tD, tH, tW = self.size

        # Scale is the fraction of the ORIGINAL image dimensions
        scale_factor = random.uniform(*self.scale)

        crop_d = max(1, int(D * scale_factor))
        crop_h = max(1, int(H * scale_factor))
        crop_w = max(1, int(W * scale_factor))

        # Random position
        d_start = random.randint(0, max(0, D - crop_d))
        h_start = random.randint(0, max(0, H - crop_h))
        w_start = random.randint(0, max(0, W - crop_w))

        volume = volume[:, d_start : d_start + crop_d, h_start : h_start + crop_h, w_start : w_start + crop_w]

        # Resize to target size using trilinear interpolation
        if (crop_d, crop_h, crop_w) != (tD, tH, tW):
            volume = torch.nn.functional.interpolate(
                volume.unsqueeze(0), size=(tD, tH, tW), mode="trilinear", align_corners=False
            ).squeeze(0)

        return volume


class RandomFlip3D:
    """Random flip along one or more axes."""

    def __init__(self, dims: list = None, p: float = 0.5):
        self.dims = dims or [-1]
        self.p = p

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        for dim in self.dims:
            if random.random() < self.p:
                volume = torch.flip(volume, [dim])
        return volume


class GaussianBlur3D:
    """3D Gaussian blur with random sigma."""

    def __init__(self, sigma: Tuple[float, float] = (0.1, 2.0), p: float = 0.5):
        self.sigma = sigma
        self.p = p

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return volume

        sigma = random.uniform(*self.sigma)
        kernel_size = max(3, int(sigma * 6) | 1)  # ensure odd
        x = torch.arange(kernel_size, dtype=volume.dtype, device=volume.device) - kernel_size // 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()

        C = volume.shape[0]
        for dim in [-3, -2, -1]:
            shape = [1, 1, 1]
            shape[dim + 3] = kernel_size
            k = kernel.reshape(*shape)
            padding = kernel_size // 2
            if dim == -3:
                v = volume.reshape(C * volume.shape[1], 1, volume.shape[2], volume.shape[3])
                v = torch.nn.functional.conv2d(v, k.reshape(1, 1, kernel_size, 1), padding=(padding, 0))
                volume = v.reshape(C, volume.shape[1], volume.shape[2], volume.shape[3])
            elif dim == -2:
                v = volume.reshape(C * volume.shape[1] * volume.shape[2], 1, volume.shape[3])
                v = torch.nn.functional.conv1d(v, k.reshape(1, 1, kernel_size), padding=padding)
                volume = v.reshape(C, volume.shape[1], volume.shape[2], volume.shape[3])
            elif dim == -1:
                v = volume.permute(0, 3, 1, 2).reshape(C * volume.shape[3], 1, volume.shape[1], volume.shape[2])
                v = torch.nn.functional.conv2d(v, k.reshape(1, 1, kernel_size, 1), padding=(padding, 0))
                volume = v.reshape(C, volume.shape[3], volume.shape[1], volume.shape[2]).permute(0, 2, 3, 1)

        return volume


class RandomRotation3D:
    """
    Random 3D rotation using explicit rotation matrices.
    Rotates around X, Y, Z axes with configurable angle ranges.
    
    The rotation matrix is constructed as R = Rz @ Ry @ Rx, where:
    - Rx rotates around X-axis
    - Ry rotates around Y-axis
    - Rz rotates around Z-axis
    """

    def __init__(
        self,
        angle_range: Tuple[float, float] = (-15.0, 15.0),
        axes: Tuple[int, int, int] = (0, 1, 2),
        p: float = 0.5,
        resample: str = "bilinear",
    ):
        """
        Args:
            angle_range: Range of rotation angles in degrees for each axis (uniform sampling).
            axes: Spatial axes to rotate around (default: all 3 axes).
            p: Probability of applying rotation.
            resample: Interpolation mode for grid_sample.
        """
        self.angle_range = tuple(angle_range)
        self.axes = tuple(axes)
        self.p = p
        self.resample = resample

    @staticmethod
    def _rotation_matrix_x(angle_deg: float) -> torch.Tensor:
        """Rotation matrix around X-axis."""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        return torch.tensor([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a],
        ], dtype=torch.float32)

    @staticmethod
    def _rotation_matrix_y(angle_deg: float) -> torch.Tensor:
        """Rotation matrix around Y-axis."""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        return torch.tensor([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a],
        ], dtype=torch.float32)

    @staticmethod
    def _rotation_matrix_z(angle_deg: float) -> torch.Tensor:
        """Rotation matrix around Z-axis."""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        return torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1],
        ], dtype=torch.float32)

    def _build_rotation_matrix(self, angles: Tuple[float, float, float]) -> torch.Tensor:
        """
        Build combined rotation matrix: R = Rz @ Ry @ Rx
        
        Args:
            angles: (angle_x, angle_y, angle_z) in degrees
        Returns:
            3x3 rotation matrix
        """
        Rx = self._rotation_matrix_x(angles[0])
        Ry = self._rotation_matrix_y(angles[1])
        Rz = self._rotation_matrix_z(angles[2])
        return Rz @ Ry @ Rx

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return volume

        C, D, H, W = volume.shape
        
        # Generate random angles for each axis
        angle_x = random.uniform(*self.angle_range)
        angle_y = random.uniform(*self.angle_range)
        angle_z = random.uniform(*self.angle_range)
        
        # Build rotation matrix
        R = self._build_rotation_matrix((angle_x, angle_y, angle_z))
        
        # Affine matrix: only rotation, no translation
        # In normalized [-1,1] coordinates, center is at origin
        affine = torch.zeros(1, 3, 4, dtype=torch.float32)
        affine[0, :3, :3] = R
        # affine[0, :3, 3] stays zero — center-rotation in normalized coords needs no translation
        
        # Create grid for 3D volume: size = (N, C, D, H, W)
        # For grid_sample with 3D input, we need to use the 3D path
        grid = torch.nn.functional.affine_grid(
            affine,
            size=[1, C, D, H, W],
            align_corners=False,
        )
        
        # grid_sample expects (N, C, D, H, W)
        volume = volume.unsqueeze(0)
        volume = torch.nn.functional.grid_sample(
            volume, grid, mode=self.resample, padding_mode="zeros", align_corners=False
        )
        volume = volume.squeeze(0)
        
        return volume


class RandomIntensityPerturbation:
    """Random intensity scaling and shifting for SPECT volumes."""

    def __init__(self, brightness: float = 0.05, contrast: float = 0.1, p: float = 0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return volume

        if self.brightness > 0:
            shift = random.uniform(-self.brightness, self.brightness)
            volume = volume + shift
            volume = torch.clamp(volume, min=0.0)

        if self.contrast > 0:
            factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            mean = volume.mean()
            volume = mean + factor * (volume - mean)
            volume = torch.clamp(volume, min=0.0)

        return volume


class DataAugmentation3D_DINO:
    """
    3D augmentation pipeline for SPECT volumes in DINOv3 self-supervised training.
    Mirrors DataAugmentationDINO but for volumetric data.
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=128,
        local_crops_size=64,
        patch_size=16,
        horizontal_flips=True,
        intensity_perturbation_p=0.5,
        rotation_angle_range=(-15.0, 15.0),
        rotation_p=0.3,
    ):
        self.global_crops_size = _make_3tuple(global_crops_size)
        self.local_crops_size = _make_3tuple(local_crops_size)
        self.local_crops_number = local_crops_number

        logger.info("###################################")
        logger.info("Using 3D data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {self.global_crops_size}")
        logger.info(f"local_crops_size: {self.local_crops_size}")
        logger.info(f"rotation_angle_range: {rotation_angle_range}")
        logger.info(f"rotation_p: {rotation_p}")
        logger.info("###################################")

        # Global crops: random crop + rotation + flip + intensity + blur
        self.global_crop = RandomCrop3D(size=self.global_crops_size, scale=global_crops_scale)
        self.global_rotate = RandomRotation3D(angle_range=rotation_angle_range, p=rotation_p)
        self.global_flip = RandomFlip3D(dims=[0, 1, 2], p=0.3)
        self.global_intensity = RandomIntensityPerturbation(
            brightness=0.05, contrast=0.1, p=intensity_perturbation_p
        )
        self.global_blur1 = GaussianBlur3D(sigma=(0.1, 2.0), p=0.2)   # always blur
        self.global_blur2 = GaussianBlur3D(sigma=(0.1, 2.0), p=0.08)   # rarely blur

        # Local crops: only create if local_crops_number > 0
        self.local_crop = None
        self.local_flip = None
        self.local_intensity = None
        self.local_blur = None
        if local_crops_number > 0:
            self.local_crop = RandomCrop3D(size=self.local_crops_size, scale=local_crops_scale)
            self.local_flip = RandomFlip3D(dims=[0, 1, 2], p=0.3)
            self.local_intensity = RandomIntensityPerturbation(
                brightness=0.05, contrast=0.1, p=intensity_perturbation_p
            )
            self.local_blur = GaussianBlur3D(sigma=(0.1, 2.0), p=0.1)

    def __call__(self, volume: torch.Tensor) -> dict:
        output = {}

        # Global crop 1 (always blurred)
        g1 = self.global_crop(volume)
        g1 = self.global_rotate(g1)
        g1 = self.global_flip(g1)
        g1 = self.global_intensity(g1)
        g1 = self.global_blur1(g1)

        # Global crop 2 (rarely blurred)
        g2 = self.global_crop(volume)
        g2 = self.global_rotate(g2)
        g2 = self.global_flip(g2)
        g2 = self.global_intensity(g2)
        g2 = self.global_blur2(g2)

        output["global_crops"] = [g1, g2]
        output["global_crops_teacher"] = [g1, g2]

        # Local crops (NO rotation — too small for 15° to be meaningful)
        local_crops = []
        for _ in range(self.local_crops_number):
            lc = self.local_crop(volume)
            lc = self.local_flip(lc)
            lc = self.local_intensity(lc)
            lc = self.local_blur(lc)
            local_crops.append(lc)

        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
