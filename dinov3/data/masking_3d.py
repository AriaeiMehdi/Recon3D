import logging
import math
import random

import numpy as np


class MaskingGenerator3D:
    """
    3D cuboid masking for iBOT-style pretraining on volumetric data.
    Generates block-wise cuboid masks on a 3D patch grid.
    """

    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 3
        self.depth, self.height, self.width = input_size

        self.num_patches = self.depth * self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator3D(%d, %d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.depth,
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.depth, self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_volume = random.uniform(self.min_num_patches, max_mask_patches)
            # Sample two aspect ratios for 3D cuboid
            aspect_dh = math.exp(random.uniform(*self.log_aspect_ratio))
            aspect_dw = math.exp(random.uniform(*self.log_aspect_ratio))

            # Compute dimensions from target volume and aspect ratios
            # d * h * w = target_volume, h/d = aspect_dh, w/d = aspect_dw
            d = int(round((target_volume / (aspect_dh * aspect_dw)) ** (1.0 / 3.0)))
            h = int(round(d * aspect_dh))
            w = int(round(d * aspect_dw))

            # Reject if cuboid doesn't fit (matching 2D behavior)
            if d < 1 or h < 1 or w < 1:
                continue
            if d > self.depth or h > self.height or w > self.width:
                continue

            if d <= self.depth and h <= self.height and w <= self.width:
                top_d = random.randint(0, self.depth - d)
                top_h = random.randint(0, self.height - h)
                top_w = random.randint(0, self.width - w)

                num_masked = mask[top_d : top_d + d, top_h : top_h + h, top_w : top_w + w].sum()
                # Overlap check
                if 0 < d * h * w - num_masked <= max_mask_patches:
                    for i in range(top_d, top_d + d):
                        for j in range(top_h, top_h + h):
                            for k in range(top_w, top_w + w):
                                if mask[i, j, k] == 0:
                                    mask[i, j, k] = 1
                                    delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return self.complete_mask_randomly(mask, num_masking_patches)

    def complete_mask_randomly(self, mask, num_masking_patches):
        shape = mask.shape
        m2 = mask.flatten()
        to_add = np.random.choice(np.where(~m2)[0], size=num_masking_patches - m2.sum(), replace=False)
        m2[to_add] = True
        return m2.reshape(shape)
