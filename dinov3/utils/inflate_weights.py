import argparse
import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger("dinov3")


def inflate_2d_to_3d(
    state_dict_2d: dict,
    model_3d: nn.Module,
    strategy: str = "repeat_depth",
) -> dict:
    """
    Inflate 2D DINOv3 weights to 3D model.

    Args:
        state_dict_2d: State dict from 2D model.
        model_3d: Target 3D model instance.
        strategy: How to inflate Conv2d weights to Conv3d.
            "repeat_depth": Repeat along depth dimension and normalize.
            "average": Repeat and average.

    Returns:
        State dict compatible with the 3D model.
    """
    state_dict_3d = model_3d.state_dict()
    inflated = {}

    for key, val_3d in state_dict_3d.items():
        if key not in state_dict_2d:
            # New parameter in 3D model (e.g., if embed_dim changed)
            logger.info(f"  [SKIP] {key}: not in 2D state dict, using 3D init")
            inflated[key] = val_3d
            continue

        val_2d = state_dict_2d[key]

        if key == "patch_embed.proj.weight":
            # Conv2d weight: (E, C, kH, kW) -> Conv3d weight: (E, C, kD, kH, kW)
            E, C, kH, kW = val_2d.shape
            kD = val_3d.shape[2]
            if strategy == "repeat_depth":
                # Repeat along depth and normalize
                w = val_2d.unsqueeze(2).repeat(1, 1, kD, 1, 1) / kD
            elif strategy == "average":
                w = val_2d.unsqueeze(2).expand(-1, -1, kD, -1, -1).clone() / kD
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            inflated[key] = w
            logger.info(f"  [INFLATE] {key}: {val_2d.shape} -> {w.shape}")

        elif key == "patch_embed.proj.bias":
            # Bias: copy unchanged
            inflated[key] = val_2d
            logger.info(f"  [COPY] {key}: {val_2d.shape}")

        elif key == "rope_embed.periods":
            # RoPE periods: 2D has D_head//4, 3D has D_head//6
            n_2d = val_2d.shape[0]
            n_3d = val_3d.shape[0]
            if n_3d <= n_2d:
                # Take first n_3d periods (covers H/W frequencies, D gets same)
                inflated[key] = val_2d[:n_3d]
            else:
                # Interpolate to larger size
                inflated[key] = torch.nn.functional.interpolate(
                    val_2d.unsqueeze(0).unsqueeze(0),
                    size=n_3d,
                    mode="linear",
                    align_corners=True,
                ).squeeze(0).squeeze(0)
            logger.info(f"  [INFLATE] {key}: {val_2d.shape} -> {inflated[key].shape}")

        elif val_2d.shape == val_3d.shape:
            # Same shape: copy directly
            inflated[key] = val_2d
            logger.info(f"  [COPY] {key}: {val_2d.shape}")

        else:
            # Shape mismatch for other parameters: use 3D init
            logger.warning(f"  [SKIP] {key}: shape mismatch 2D={val_2d.shape} vs 3D={val_3d.shape}")
            inflated[key] = val_3d

    return inflated


def inflate_checkpoint(input_path: str, output_path: str, model_name: str = "vit_small_3d"):
    """
    Load a 2D checkpoint and inflate it to a 3D model checkpoint.

    Args:
        input_path: Path to 2D checkpoint file.
        output_path: Path to save inflated 3D checkpoint.
        model_name: Name of the 3D model architecture to use.
    """
    from dinov3.models import vision_transformer_3d as vits3d

    logger.info(f"Loading 2D checkpoint from: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "teacher" in checkpoint:
        state_dict_2d = checkpoint["teacher"]
        logger.info("Found 'teacher' key in checkpoint")
    elif "model" in checkpoint:
        state_dict_2d = checkpoint["model"]
        logger.info("Found 'model' key in checkpoint")
    elif "state_dict" in checkpoint:
        state_dict_2d = checkpoint["state_dict"]
        logger.info("Found 'state_dict' key in checkpoint")
    else:
        state_dict_2d = checkpoint
        logger.info("Using checkpoint directly as state dict")

    # Remove "module." prefix if present
    state_dict_2d = {k.removeprefix("module."): v for k, v in state_dict_2d.items()}

    # Infer 2D model params from state dict
    embed_dim = state_dict_2d["patch_embed.proj.weight"].shape[0]
    patch_size_hw = state_dict_2d["patch_embed.proj.weight"].shape[2]
    num_blocks = sum(1 for k in state_dict_2d if k.startswith("blocks.") and k.endswith(".norm1.weight"))

    logger.info(f"Detected 2D model: embed_dim={embed_dim}, patch_size={patch_size_hw}, depth={num_blocks}")

    # Build matching 3D model
    model_3d = vits3d.__dict__[model_name](
        patch_size=patch_size_hw,
        embed_dim=embed_dim,
    )
    logger.info(f"Created 3D model: {model_name}")

    # Inflate weights
    logger.info("Inflating weights...")
    inflated_state_dict = inflate_2d_to_3d(state_dict_2d, model_3d)

    # Save
    logger.info(f"Saving inflated checkpoint to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(inflated_state_dict, output_path)
    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(description="Inflate 2D DINOv3 weights to 3D")
    parser.add_argument("--input", type=str, required=True, help="Path to 2D checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to save 3D checkpoint")
    parser.add_argument(
        "--model",
        type=str,
        default="vit_small_3d",
        choices=["vit_small_3d", "vit_base_3d", "vit_large_3d"],
        help="3D model architecture",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    inflate_checkpoint(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
