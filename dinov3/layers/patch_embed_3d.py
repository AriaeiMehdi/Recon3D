import math
from typing import Callable, Tuple, Union

from torch import Tensor, nn


def make_3tuple(x):
    if hasattr(x, '__iter__') and not isinstance(x, str):
        x = list(x)
        assert len(x) == 3
        return tuple(x)

    assert isinstance(x, int)
    return (x, x, x)


class PatchEmbed3D(nn.Module):
    """
    3D volume to patch embedding: (B,C,D,H,W) -> (B,N,D) or (B,D',H',W',D)

    Args:
        img_size: Volume size.
        patch_size: Patch token size.
        in_chans: Number of input channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
        flatten_embedding: If False, return (B, D', H', W', embed_dim).
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]] = 128,
        patch_size: Union[int, Tuple[int, int, int]] = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_DHW = make_3tuple(img_size)
        patch_DHW = make_3tuple(patch_size)
        patch_grid_size = (
            image_DHW[0] // patch_DHW[0],
            image_DHW[1] // patch_DHW[1],
            image_DHW[2] // patch_DHW[2],
        )

        self.img_size = image_DHW
        self.patch_size = patch_DHW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1] * patch_grid_size[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_DHW, stride=patch_DHW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, D, H, W = x.shape

        x = self.proj(x)  # B C D H W
        D, H, W = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)  # B DHW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, D, H, W, self.embed_dim)  # B D H W C
        return x

    def flops(self) -> float:
        Do, Ho, Wo = self.patches_resolution
        flops = Do * Ho * Wo * self.embed_dim * self.in_chans * (
            self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        )
        if self.norm is not None:
            flops += Do * Ho * Wo * self.embed_dim
        return flops

    def reset_parameters(self):
        k = 1 / (self.in_chans * self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))
