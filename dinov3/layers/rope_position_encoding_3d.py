import math
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn


class RopePositionEmbedding3D(nn.Module):
    """
    3D axial RoPE: separate rotation for D, H, W axes.
    Mirrors RopePositionEmbedding with 3 spatial dimensions.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        # For 3D axial: 3 axes * D_head//6 periods per axis = D_head//2 total angle dims
        # D_head // 6 periods per axis, remainder dims get cos=1, sin=0
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.n_periods = D_head // 6  # periods per axis for 3D
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        self.dtype = dtype
        self.register_buffer(
            "periods",
            torch.empty(self.n_periods, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, D: int, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_DHW = max(D, H, W)
            coords_d = torch.arange(0.5, D, **dd) / max_DHW
            coords_h = torch.arange(0.5, H, **dd) / max_DHW
            coords_w = torch.arange(0.5, W, **dd) / max_DHW
        elif self.normalize_coords == "min":
            min_DHW = min(D, H, W)
            coords_d = torch.arange(0.5, D, **dd) / min_DHW
            coords_h = torch.arange(0.5, H, **dd) / min_DHW
            coords_w = torch.arange(0.5, W, **dd) / min_DHW
        elif self.normalize_coords == "separate":
            coords_d = torch.arange(0.5, D, **dd) / D
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        coords = torch.stack(
            torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"), dim=-1
        )  # [D, H, W, 3]
        coords = coords.flatten(0, 2)  # [DHW, 3]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_dhw = torch.empty(3, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_dhw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_dhw = torch.empty(3, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_dhw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_dhw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_dhw

        # Prepare angles and sin/cos
        # coords: [DHW, 3], periods: [n_periods]
        # angles: [DHW, 3, n_periods] -> flatten to [DHW, 3*n_periods]
        n_periods = self.n_periods
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [DHW, 3, n_periods]
        angles = angles.flatten(1, 2)  # [DHW, 3*n_periods]

        # Pad or tile to fill D_head dimensions
        angle_dim = angles.shape[-1]  # 3 * n_periods
        if angle_dim * 2 < self.D_head:
            # Pad with zeros for remaining dims (those dims get cos=1, sin=0)
            pad_size = self.D_head - angle_dim * 2
            angles = torch.cat([angles, torch.zeros(angles.shape[0], pad_size, **dd)], dim=1)
            angles = torch.cat([angles, angles[:, :angle_dim]], dim=1)  # tile the first half
        else:
            angles = angles[:, : self.D_head // 2]
            angles = angles.tile(2)  # [DHW, D_head]

        angles = angles[:, : self.D_head]  # ensure exact size
        cos = torch.cos(angles)  # [DHW, D_head]
        sin = torch.sin(angles)  # [DHW, D_head]

        return (sin, cos)

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        n = self.n_periods
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(n, device=device, dtype=dtype) / (2 * n)
            )  # [n_periods] — exponents span [0, ~1.0) matching 2D behavior
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, n, device=device, dtype=dtype)
            periods = base**exponents
            periods = periods / base
            periods = periods * self.max_period
        self.periods.data = periods
