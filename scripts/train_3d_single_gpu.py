import os
# Prevent CPU thread explosion in DataLoader workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


import argparse
import logging
import math
import sys
import time

import torch
import torch.nn.functional as F
from functools import partial
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dinov3.data.loaders import make_data_loader, SamplerType
from dinov3.data.augmentations_3d import DataAugmentation3D_DINO
from dinov3.data.collate_3d import collate_data_and_cast_3d
from dinov3.data.masking_3d import MaskingGenerator3D
from dinov3.layers.dino_head import DINOHead
from dinov3.loss.dino_clstoken_loss import DINOLoss
from dinov3.loss.ibot_patch_loss import iBOTPatchLoss
from dinov3.loss.koleo_loss import KoLeoLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("dinov3")


def setup_logging(output_dir):
    """Setup logging to both console and file."""
    log_file = os.path.join(output_dir, "train_log.txt")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.info(f"Logging to {log_file}")
    return log_file


def build_model_3d(cfg):
    from dinov3.models.vision_transformer_3d import DinoVisionTransformer3D
    model = DinoVisionTransformer3D(
        img_size=tuple(cfg.crops.global_crops_size),
        patch_size=cfg.student.patch_size,
        in_chans=cfg.student.in_chans,
        embed_dim=cfg.student.embed_dim,
        depth=cfg.student.depth,
        num_heads=cfg.student.num_heads,
        ffn_ratio=cfg.student.ffn_ratio,
        pos_embed_rope_base=cfg.student.pos_embed_rope_base,
        pos_embed_rope_normalize_coords=cfg.student.pos_embed_rope_normalize_coords,
        pos_embed_rope_dtype=cfg.student.pos_embed_rope_dtype,
        qkv_bias=cfg.student.qkv_bias,
        layerscale_init=cfg.student.layerscale,
        norm_layer=cfg.student.norm_layer,
        ffn_layer=cfg.student.ffn_layer,
        ffn_bias=cfg.student.ffn_bias,
        proj_bias=cfg.student.proj_bias,
        n_storage_tokens=cfg.student.n_storage_tokens,
        mask_k_bias=cfg.student.mask_k_bias,
        drop_path_rate=cfg.student.drop_path_rate,
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./output_3d")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    opts = parser.parse_args()

    cfg = OmegaConf.load(opts.config_file)
    OmegaConf.set_struct(cfg, False)
    cfg.train.output_dir = opts.output_dir
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    setup_logging(cfg.train.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Patch SinkhornKnoppTeacher to work without distributed (single GPU)
    from dinov3.loss.ibot_patch_loss import SinkhornKnoppTeacher
    def _single_gpu_sk_forward(self, teacher_output, teacher_temp, n_masked_patches_tensor=None, n_iterations=3):
        teacher_output = teacher_output.float()
        Q = torch.exp(teacher_output / teacher_temp).t()  # [K, B]
        B = Q.shape[1] if n_masked_patches_tensor is None else n_masked_patches_tensor
        K = Q.shape[0]

        # Normalize total probability mass to 1
        Q /= torch.sum(Q)

        # Sinkhorn iterations with correct marginal normalization
        for _ in range(n_iterations):
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= K                                   # row marginal = 1/K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B                                   # column marginal = 1/B

        Q *= B
        return Q.t()
    SinkhornKnoppTeacher.forward = _single_gpu_sk_forward

    # Patch DINOLoss.reduce_center_update for single GPU (skip dist.all_reduce)
    from dinov3.loss.dino_clstoken_loss import DINOLoss
    def _single_gpu_reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # Skip dist.all_reduce — single GPU, no need to aggregate across ranks
    DINOLoss.reduce_center_update = _single_gpu_reduce_center_update

    # Patch iBOTPatchLoss.reduce_center_update for single GPU (same fix)
    from dinov3.loss.ibot_patch_loss import iBOTPatchLoss
    def _single_gpu_ibot_reduce_center_update(self, teacher_patch_tokens):
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(teacher_patch_tokens.mean(1), dim=0, keepdim=True)
        # Skip dist.all_reduce — single GPU
    iBOTPatchLoss.reduce_center_update = _single_gpu_ibot_reduce_center_update

    # Build student and teacher
    logger.info("Building models...")
    student = build_model_3d(cfg).to(device)
    teacher = build_model_3d(cfg).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    embed_dim = student.embed_dim
    logger.info(f"embed_dim={embed_dim}, params={sum(p.numel() for p in student.parameters()) / 1e6:.1f}M")

    # Build heads — student and teacher each get their own copy
    # (matches ssl_meta_arch.py lines 82-83, 126-127)
    dino_head_fn = partial(
        DINOHead,
        in_dim=embed_dim,
        out_dim=cfg.dino.head_n_prototypes,
        hidden_dim=cfg.dino.head_hidden_dim,
        bottleneck_dim=cfg.dino.head_bottleneck_dim,
        nlayers=cfg.dino.head_nlayers,
    )

    student_dino_head = dino_head_fn().to(device)
    teacher_dino_head = dino_head_fn().to(device)
    teacher_dino_head.load_state_dict(student_dino_head.state_dict())
    for p in teacher_dino_head.parameters():
        p.requires_grad = False

    if cfg.ibot.separate_head:
        ibot_head_fn = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.ibot.head_n_prototypes,
            hidden_dim=cfg.ibot.head_hidden_dim,
            bottleneck_dim=cfg.ibot.head_bottleneck_dim,
            nlayers=cfg.ibot.head_nlayers,
        )
        student_ibot_head = ibot_head_fn().to(device)
        teacher_ibot_head = ibot_head_fn().to(device)
        teacher_ibot_head.load_state_dict(student_ibot_head.state_dict())
        for p in teacher_ibot_head.parameters():
            p.requires_grad = False
    else:
        student_ibot_head = student_dino_head
        teacher_ibot_head = teacher_dino_head

    # Losses
    # Use slower center momentum to prevent center collapse with small batches
    # Default 0.9 tracks teacher too closely → teacher_output - center ≈ 0 → uniform
    dino_center_momentum = getattr(cfg.dino, "center_momentum", 0.99)
    dino_loss = DINOLoss(cfg.dino.head_n_prototypes, center_momentum=dino_center_momentum).to(device)
    dino_loss.init_weights()  # Initialize center to zeros (starts as NaN)
    ibot_center_momentum = getattr(cfg.ibot, "center_momentum", 0.999)
    ibot_loss = iBOTPatchLoss(cfg.ibot.head_n_prototypes, center_momentum=ibot_center_momentum).to(device)
    ibot_loss.init_weights()  # Initialize center to zeros (starts as NaN)
    # Disable torch.compile on sinkhorn_knopp (needs triton)
    ibot_loss.sinkhorn_knopp_teacher = ibot_loss.sinkhorn_knopp_teacher.__class__()
    ibot_loss.sinkhorn_knopp_teacher.to(device)
    koleo_loss = KoLeoLoss().to(device) if cfg.dino.koleo_loss_weight > 0 else None

    # Data augmentation
    augmentation = DataAugmentation3D_DINO(
        global_crops_scale=cfg.crops.global_crops_scale,
        local_crops_scale=cfg.crops.local_crops_scale,
        local_crops_number=cfg.crops.local_crops_number,
        global_crops_size=tuple(cfg.crops.global_crops_size),
        local_crops_size=tuple(cfg.crops.local_crops_size),
        patch_size=cfg.student.patch_size,
        rotation_angle_range=tuple(cfg.crops.rotation_angle_range),
        rotation_p=cfg.crops.rotation_p,
    )

    # Dataset and loader
    from dinov3.data.datasets.spect import SPECT3D
    dataset = SPECT3D(
        root=cfg.train.dataset_path,
        transform=augmentation,
        normalize=getattr(cfg.train, "normalize", "none"),
    )

    # Masking
    img_size = list(cfg.crops.global_crops_size)
    patch_size = cfg.student.patch_size
    if isinstance(img_size, list) and len(img_size) == 3:
        grid_d, grid_h, grid_w = img_size[0] // patch_size, img_size[1] // patch_size, img_size[2] // patch_size
    else:
        grid_d = grid_h = grid_w = img_size // patch_size
    n_tokens = grid_d * grid_h * grid_w
    mask_generator = MaskingGenerator3D(
        input_size=(grid_d, grid_h, grid_w),
        max_num_patches=int(0.5 * n_tokens),
    )

    collate_fn = partial(
        collate_data_and_cast_3d,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        dtype=torch.float32,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
    )

    # Patient-aware sampler
    from dinov3.data.patient_sampler import PatientAwareInfiniteSampler, extract_patient_id

    patient_ids = [extract_patient_id(p) for p in dataset.samples]
    sampler = PatientAwareInfiniteSampler(patient_ids, cfg.train.batch_size_per_gpu, seed=cfg.train.seed)
    logger.info(f"sampler: patient_aware_infinite (batch_size={cfg.train.batch_size_per_gpu})")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False,
        prefetch_factor=2,
    )

    # Optimizer — only student params get gradients
    param_groups = [
        {"params": [p for n, p in student.named_parameters() if "patch_embed" in n], "lr": cfg.optim.lr * cfg.optim.patch_embed_lr_mult, "name": "patch_embed"},
        {"params": [p for n, p in student.named_parameters() if "patch_embed" not in n], "lr": cfg.optim.lr, "name": "backbone"},
        {"params": list(student_dino_head.parameters()), "lr": cfg.optim.lr, "name": "dino_head"},
    ]
    if student_ibot_head is not student_dino_head:
        param_groups.append({"params": list(student_ibot_head.parameters()), "lr": cfg.optim.lr, "name": "ibot_head"})
    if koleo_loss is not None:
        param_groups.append({"params": list(koleo_loss.parameters()), "lr": cfg.optim.lr, "name": "koleo"})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.optim.weight_decay, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))

    # Test patient-aware sampler
    def test_patient_aware_sampler(dataset, sampler, batch_size, n_batches_to_check=200):
        sampler_iter = iter(sampler)
        patient_ids = [extract_patient_id(p) for p in dataset.samples]

        total_batches = 0
        collision_batches = 0
        max_collisions = 0
        example_collision = None

        for _ in range(n_batches_to_check):
            batch_indices = [next(sampler_iter) for _ in range(batch_size)]
            batch_patients = [patient_ids[i] for i in batch_indices]
            seen = set()
            collisions = 0
            for pid in batch_patients:
                if pid in seen:
                    collisions += 1
                    if example_collision is None:
                        example_collision = (pid, batch_patients)
                seen.add(pid)
            total_batches += 1
            if collisions > 0:
                collision_batches += 1
            max_collisions = max(max_collisions, collisions)

        rate = collision_batches / total_batches
        logger.info(f"PatientAwareSampler test ({n_batches_to_check} batches):")
        logger.info(f"  Batches with >=1 collision: {collision_batches}/{total_batches} ({100*rate:.1f}%)")
        logger.info(f"  Max collisions in a single batch: {max_collisions}")
        if example_collision:
            logger.warning(f"  Example collision: patient {example_collision[0]} appeared twice")
        else:
            logger.info(f"  No collisions detected.")

    test_patient_aware_sampler(dataset, sampler, cfg.train.batch_size_per_gpu)

    # Resume from checkpoint
    start_iteration = 0
    if opts.resume:
        import glob
        ckpt_files = sorted(glob.glob(os.path.join(cfg.train.output_dir, "checkpoint_*.pt")))
        if ckpt_files:
            latest_ckpt = ckpt_files[-1]
            logger.info(f"Resuming from: {latest_ckpt}")
            ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
            student.load_state_dict(ckpt["student"])
            teacher.load_state_dict(ckpt["teacher"])
            student_dino_head.load_state_dict(ckpt["student_dino_head"])
            teacher_dino_head.load_state_dict(ckpt["teacher_dino_head"])
            if student_ibot_head is not student_dino_head:
                student_ibot_head.load_state_dict(ckpt["student_ibot_head"])
                teacher_ibot_head.load_state_dict(ckpt["teacher_ibot_head"])
            optimizer.load_state_dict(ckpt["optimizer"])
            dino_loss.load_state_dict(ckpt["dino_loss"])
            ibot_loss.load_state_dict(ckpt["ibot_loss"])
            start_iteration = ckpt["iteration"] + 1
            logger.info(f"Resumed at iteration {start_iteration}")
        else:
            logger.info("No checkpoints found, starting from scratch")

    # Training loop
    total_iter = cfg.optim.epochs * cfg.train.OFFICIAL_EPOCH_LENGTH
    warmup_iter = cfg.optim.warmup_epochs * cfg.train.OFFICIAL_EPOCH_LENGTH
    base_lr = cfg.optim.lr
    logger.info(f"Training for {total_iter} iterations ({cfg.optim.epochs} epochs x {cfg.train.OFFICIAL_EPOCH_LENGTH} steps/epoch)")

    data_iter = iter(data_loader)
    t0 = time.time()
    grad_accum_steps = getattr(cfg.optim, "gradient_accumulation_steps", 1)
    optimizer.zero_grad()
    t_cls_accum = []

    for iteration in range(start_iteration, total_iter):
        # LR schedule: cosine decay from base_lr to min_lr (no warmup)
        progress = iteration / max(total_iter, 1)
        lr = cfg.optim.min_lr + 0.5 * (base_lr - cfg.optim.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            mult = cfg.optim.patch_embed_lr_mult if pg.get("name") == "patch_embed" else 1.0
            pg["lr"] = lr * mult

        # Teacher momentum schedule
        mom = cfg.teacher.final_momentum_teacher - (cfg.teacher.final_momentum_teacher - cfg.teacher.momentum_teacher) * (
            1 - iteration / total_iter
        )

        # Teacher temperature
        if iteration < cfg.teacher.warmup_teacher_temp_epochs * cfg.train.OFFICIAL_EPOCH_LENGTH:
            teacher_temp = cfg.teacher.warmup_teacher_temp + (cfg.teacher.teacher_temp - cfg.teacher.warmup_teacher_temp) * iteration / (
                cfg.teacher.warmup_teacher_temp_epochs * cfg.train.OFFICIAL_EPOCH_LENGTH
            )
        else:
            teacher_temp = cfg.teacher.teacher_temp

        # Get data
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            data = next(data_iter)

        global_crops = data["collated_global_crops"].to(device)
        local_crops = data.get("collated_local_crops")
        if local_crops is not None:
            local_crops = local_crops.to(device)
        masks = data["collated_masks"].to(device)
        mask_indices_list = data["mask_indices_list"].to(device)
        masks_weight = data["masks_weight"].to(device)
        n_masked_patches = data["n_masked_patches"].to(device)

        n_global = 2
        n_local = cfg.crops.local_crops_number
        B = global_crops.shape[0] // n_global

        # Teacher forward (no grad) — uses teacher's own heads
        with torch.no_grad():
            teacher_out = teacher(global_crops, is_training=True)
            t_cls = teacher_dino_head(teacher_out["x_norm_clstoken"])
            t_patch = teacher_ibot_head(teacher_out["x_norm_patchtokens"].flatten(0, 1))
            t_cls_accum.append(t_cls.clone())

            # DINO cls-token centering: softmax center (DINOv1/v2 style)
            # Use EMA center update — NOT Sinkhorn (SK breaks CLS batch statistics)
            # Respects cfg.train.centering: "softmax_center" (default) or "sinkhorn_knopp"
            dino_centering = getattr(cfg.train, "centering", "softmax_center")
            if dino_centering == "sinkhorn_knopp":
                t_cls_centered = dino_loss.sinkhorn_knopp_teacher(t_cls, teacher_temp=teacher_temp)
            else:
                t_cls_centered = dino_loss.softmax_center_teacher(t_cls, teacher_temp=teacher_temp)

            # iBOT patch centering: SK (balanced assignment, batch-size independent)
            t_patch_centered = ibot_loss.sinkhorn_knopp_teacher(
                torch.index_select(t_patch, 0, mask_indices_list),
                teacher_temp=teacher_temp,
                n_masked_patches_tensor=n_masked_patches,
            )

        # Student forward
        student_inputs = [global_crops]
        student_masks = [masks]
        if n_local > 0 and local_crops is not None:
            student_inputs.append(local_crops)
            student_masks.append(None)

        student_out = student(
            student_inputs,
            masks=student_masks,
            is_training=True,
        )

        s_cls_global = student_dino_head(student_out[0]["x_norm_clstoken"])
        s_patch_global = student_ibot_head(student_out[0]["x_norm_patchtokens"].flatten(0, 1))

        if n_local > 0:
            s_cls_local = student_dino_head(student_out[1]["x_norm_clstoken"])
        else:
            s_cls_local = None

        # Select only masked patches for student
        s_patch_masked = torch.index_select(s_patch_global, 0, mask_indices_list)

        # DINO loss
        s_cls_global_un = s_cls_global.unflatten(0, (n_global, B))
        t_cls_centered_un = t_cls_centered.unflatten(0, (n_global, B))

        loss_dino_global = dino_loss(s_cls_global_un, t_cls_centered_un, ignore_diagonal=True)

        if n_local > 0:
            s_cls_local_un = s_cls_local.unflatten(0, (n_local, B))
            loss_dino_local = dino_loss(s_cls_local_un, t_cls_centered_un)

            dino_global_terms = n_global * (n_global - 1)
            dino_local_terms = n_global * n_local
            dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
            dino_local_scale = dino_local_terms / (dino_global_terms + dino_local_terms)
            loss_dino = dino_global_scale * loss_dino_global + dino_local_scale * loss_dino_local
        else:
            loss_dino_local = torch.tensor(0.0, device=device)
            loss_dino = loss_dino_global

        # iBOT loss — match original: n_masked_patches=mask_indices_list.shape[0]
        loss_ibot = ibot_loss.forward_masked(
            s_patch_masked, t_patch_centered, masks,
            n_masked_patches=mask_indices_list.shape[0], masks_weight=masks_weight
        )
        # KoLeo loss — match original: per-crop, then average
        koleo_scale = n_global
        if koleo_loss is not None:
            loss_koleo = sum(koleo_loss(x) for x in student_out[0]["x_norm_clstoken"].unflatten(0, (n_global, B))) / n_global
        else:
            loss_koleo = 0

        total_loss = cfg.dino.loss_weight * loss_dino + cfg.ibot.loss_weight * loss_ibot + cfg.dino.koleo_loss_weight * koleo_scale * loss_koleo

        # Scale loss for gradient accumulation
        scaled_loss = total_loss / grad_accum_steps
        scaled_loss.backward()

        # Step optimizer + EMA update only after accumulating enough gradients
        if (iteration + 1) % grad_accum_steps == 0:
            all_trainable = list(student.parameters()) + list(student_dino_head.parameters())
            if student_ibot_head is not student_dino_head:
                all_trainable += list(student_ibot_head.parameters())
            torch.nn.utils.clip_grad_norm_(all_trainable, cfg.optim.clip_grad)
            optimizer.step()
            optimizer.zero_grad()

            # EMA update — update teacher backbone + teacher heads
            with torch.no_grad():
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.data.mul_(mom).add_(ps.data, alpha=1 - mom)
                for ps, pt in zip(student_dino_head.parameters(), teacher_dino_head.parameters()):
                    pt.data.mul_(mom).add_(ps.data, alpha=1 - mom)
                if student_ibot_head is not student_dino_head:
                    for ps, pt in zip(student_ibot_head.parameters(), teacher_ibot_head.parameters()):
                        pt.data.mul_(mom).add_(ps.data, alpha=1 - mom)

            # DINO center: update with accumulated micro-batches
            all_t_cls = torch.cat(t_cls_accum, dim=0)
            dino_loss.update_center(all_t_cls)
            t_cls_accum.clear()

        if iteration % 50 == 0:
            elapsed = time.time() - t0
            # Diagnostic: cls token std for collapse detection
            teacher_cls_std = teacher_out["x_norm_clstoken"].std(dim=0).mean().item()
            student_cls_std = student_out[0]["x_norm_clstoken"].std(dim=0).mean().item()
            dino_diff = abs(loss_dino_global.item() - loss_dino_local.item()) if n_local > 0 else 0.0
            logger.info(
                f"[{iteration}/{total_iter}] loss={total_loss.item():.4f} "
                f"dino_g={loss_dino_global.item():.6f} dino_l={loss_dino_local.item():.6f} "
                f"dino_diff={dino_diff:.2e} "
                f"ibot={loss_ibot.item():.4f} "
                f"lr={lr:.6f} mom={mom:.4f} time={elapsed:.1f}s "
                f"t_cls_std={teacher_cls_std:.4f} s_cls_std={student_cls_std:.4f}"
            )

        if (iteration + 1) % cfg.checkpointing.period == 0:
            ckpt_path = os.path.join(cfg.train.output_dir, f"checkpoint_{iteration+1}.pt")
            torch.save({
                "iteration": iteration,
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "student_dino_head": student_dino_head.state_dict(),
                "teacher_dino_head": teacher_dino_head.state_dict(),
                "student_ibot_head": student_ibot_head.state_dict() if student_ibot_head is not student_dino_head else student_dino_head.state_dict(),
                "teacher_ibot_head": teacher_ibot_head.state_dict() if student_ibot_head is not student_dino_head else teacher_dino_head.state_dict(),
                "dino_loss": dino_loss.state_dict(),
                "ibot_loss": ibot_loss.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # Final save
    ckpt_path = os.path.join(cfg.train.output_dir, "checkpoint_final.pt")
    torch.save({
        "iteration": iteration,
        "student": student.state_dict(),
        "teacher": teacher.state_dict(),
        "student_dino_head": student_dino_head.state_dict(),
        "teacher_dino_head": teacher_dino_head.state_dict(),
        "student_ibot_head": student_ibot_head.state_dict() if student_ibot_head is not student_dino_head else student_dino_head.state_dict(),
        "teacher_ibot_head": teacher_ibot_head.state_dict() if student_ibot_head is not student_dino_head else teacher_dino_head.state_dict(),
        "dino_loss": dino_loss.state_dict(),
        "ibot_loss": ibot_loss.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, ckpt_path)
    logger.info(f"Training complete. Final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
