"""adv-NO (L1 + perceptual + adversarial) fix-window training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.train.adv_no_models import (
    build_perceptual_loss,
    perceptual_backend_label,
    UNetDiscriminatorSN2D,
)
from src.train.fix_window_train import (
    build_optimizer,
    SimulationDataset,
    TrainConfig,
    _atomic_torch_save,
    _index_tensors,
    _paths,
    _save_train_state,
    build_model,
    load_merged_training_arrays,
    load_training_arrays,
    prepare_train_val_tensors,
    split_train_val_indices,
)
from src.train.fix_window_viz import (
    circle_l1_hist_key,
    compute_viz_sample_circle_l1,
    display_adv_no_epoch_plots,
    ensure_circle_l1_hist,
    fixed_viz_indices,
    save_adv_no_training_curves,
    save_fix_window_validation_figures,
    save_gaussian_circle_l1_curves,
    save_gaussian_circles_validation_figures,
    save_loss_mask_preview,
    save_validation_mask_preview,
)
from src.train.losses import (
    AdvNOLoss,
    build_loss_spatial_weight,
    expand_invalid_mask_for_loss,
    gaussian_weight_at_radius,
    gaussian_weight_ring_summary,
    gaussian_weight_summary,
    reconstruction_loss,
)


@dataclass
class AdvNOTrainConfig(TrainConfig):
    """Extends baseline TrainConfig with adv-NO hyperparameters."""

    loss_mode: str = "adv_no"  # "l1" | "adv_no"
    beta: float = 0.1
    lambda_perc: float = 0.1
    d_lr: float = 1e-4  # used only when d_lr_follow_g=False
    d_lr_follow_g: bool = True  # keep D LR = G LR * d_lr_ratio each epoch
    d_lr_ratio: float = 0.5  # d_lr = g_lr / 2 when follow_g
    d_skip_threshold: float = 0.2  # skip D update when batch D loss is below (0 = off)
    use_cosine_lr: bool = True  # cosine decay for G; else ReduceLROnPlateau on val L1
    cosine_eta_min: float | None = None  # default: lr * 0.01
    perceptual_warmup_epochs: int = 5  # L1 only for epochs <= this; perceptual from epoch+1
    warmup_epochs: int = 5  # no adversarial for epochs <= this; adv from epoch+1
    perceptual_backend: str = "vgg"  # "vgg" | "lpips"
    lpips_net: str = "alex"  # alex | vgg | squeeze (when perceptual_backend == "lpips")
    perceptual_weights: tuple[float, ...] = (0.1, 0.1, 1.0, 1.0, 1.0)
    checkpoint_subdir: str = "Models/adv_no"
    isolate_runs: bool = True  # save under checkpoint_subdir/run_name/ (no cross-run overwrite)
    best_model_path: str = ""  # default: {run_dir}/best_model.pth
    viz_every: int = 1
    viz_samples: int = 4
    viz_seed: int = 42
    viz_dir: str | None = None
    viz_residual_metric: str = "l1"
    channel_names: tuple[str, ...] = ("u", "v", "ssh")
    loss_metric: str = "components"  # "components" | "speed" (L1 on |U,V| from u,v)
    val_reconstruction_metric: str = "rmse"  # "l1" | "rmse" — metric for val loss / best checkpoint
    loss_spatial_mode: str = "full"  # "full" | "gaussian_center"
    loss_gaussian_sigma: float = 32.0  # σ in pixels when loss_spatial_mode == "gaussian_center"
    protect_existing_run: bool = True  # refuse to overwrite a completed / in-progress run dir
    preview_loss_mask: bool = True  # save loss-mask figure under viz/ before epoch 1
    cache_pred_tensors: bool = True  # save viz/pred_cache/ each viz epoch for post-hoc plots
    checkpoint_every_epochs: int = 10  # 0=off; save checkpoints/epoch_XXX.pth for replay
    show_inline_plots: bool = False
    viz_gaussian_circles: bool = False  # also save samples_gauss_circles/ each viz epoch
    viz_gaussian_circle_radii: tuple[float, ...] = (16.0, 32.0, 64.0)
    viz_gaussian_inner_circle_radii: tuple[float, ...] = (32.0, 16.0)  # zoom error cols (0 = off)
    viz_gaussian_fig_scale: float = 2.6

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdvNOTrainConfig:
        allowed = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in allowed}
        if "unet_dim_mults" in values and isinstance(values["unet_dim_mults"], list):
            values["unet_dim_mults"] = tuple(values["unet_dim_mults"])
        if "perceptual_weights" in values and isinstance(values["perceptual_weights"], list):
            values["perceptual_weights"] = tuple(values["perceptual_weights"])
        if "channel_names" in values and isinstance(values["channel_names"], list):
            values["channel_names"] = tuple(values["channel_names"])
        if "viz_gaussian_circle_radii" in values and isinstance(values["viz_gaussian_circle_radii"], list):
            values["viz_gaussian_circle_radii"] = tuple(values["viz_gaussian_circle_radii"])
        if "viz_gaussian_inner_circle_radii" in values and isinstance(
            values["viz_gaussian_inner_circle_radii"], list,
        ):
            values["viz_gaussian_inner_circle_radii"] = tuple(values["viz_gaussian_inner_circle_radii"])
        if "viz_gaussian_inner_circle_radius" in values:
            legacy = values.pop("viz_gaussian_inner_circle_radius")
            if "viz_gaussian_inner_circle_radii" not in values and legacy:
                values["viz_gaussian_inner_circle_radii"] = (float(legacy), 16.0)
        return cls(**values)

    def run_dir(self) -> Path:
        base = Path(self.checkpoint_subdir)
        return base / self.run_name if self.isolate_runs else base


def _resolve_best_model_path(config: AdvNOTrainConfig) -> Path:
    if config.best_model_path:
        return Path(config.best_model_path)
    return config.run_dir() / "best_model.pth"


def _run_has_artifacts(run_dir: Path, run_name: str) -> bool:
    """True if this run directory already holds training outputs."""
    if not run_dir.exists():
        return False
    candidates = [
        run_dir / "best_model.pth",
        run_dir / f"last_model_{run_name}.pth",
        run_dir / "history" / f"training_history_{run_name}.pt",
        run_dir / f"train_state_{run_name}.json",
        run_dir / "run_info.pt",
    ]
    for path in candidates:
        if path.is_file():
            return True
    samples = run_dir / "viz" / "samples"
    if samples.is_dir() and any(samples.iterdir()):
        return True
    gauss = run_dir / "viz" / "samples_gauss_circles"
    if gauss.is_dir() and any(gauss.iterdir()):
        return True
    return False


def _clear_run_artifacts(run_dir: Path, run_name: str) -> None:
    """Remove prior training outputs so a rerun with the same RUN_NAME starts fresh."""
    files = [
        run_dir / "best_model.pth",
        run_dir / f"last_model_{run_name}.pth",
        run_dir / "history" / f"training_history_{run_name}.pt",
        run_dir / f"train_state_{run_name}.json",
        run_dir / f"resume_{run_name}.pt",
        run_dir / "run_info.pt",
    ]
    for path in files:
        if path.is_file():
            path.unlink()

    for sub in ("samples", "samples_gauss_circles", "pred_cache", "curves"):
        viz_sub = run_dir / "viz" / sub
        if viz_sub.is_dir():
            for child in viz_sub.iterdir():
                if child.is_file():
                    child.unlink()

    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        for child in ckpt_dir.iterdir():
            if child.is_file():
                child.unlink()


def _prepare_run_dir(config: AdvNOTrainConfig) -> Path:
    """Create run directory; never delete a previous run with the same name."""
    run_dir = config.run_dir()
    if _run_has_artifacts(run_dir, config.run_name):
        if config.protect_existing_run:
            raise FileExistsError(
                f"Run already exists: {run_dir}\n"
                "Choose a new RUN_NAME / RUN_TAG so each experiment is stored separately. "
                "Set protect_existing_run=False to overwrite and restart training in this folder."
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _adv_paths(config: AdvNOTrainConfig):
    run_dir = _prepare_run_dir(config)
    config.checkpoint_dir = str(run_dir)
    return _paths(config)


def _default_adv_hist() -> dict:
    return {
        "epoch": [],
        "train": [],
        "train_l1": [],
        "train_perceptual": [],
        "train_adv": [],
        "train_d": [],
        "val": [],
        "val_l1": [],
        "val_rmse": [],
        "val_perceptual": [],
        "lr": [],
        "d_lr": [],
        "best_val_loss": float("inf"),
        "best_epoch": 0,
    }


def _try_resume_adv_no(
    config: AdvNOTrainConfig,
    model: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    scheduler_g: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    checkpoint_dir: Path,
    history_path: Path,
    resume_path: Path,
) -> tuple[int, dict, float, int]:
    """Resume from resume checkpoint, or history + last_model with optimizer reset."""
    start_epoch = 1
    hist = _default_adv_hist()
    best_val_loss = float("inf")
    best_epoch = 0

    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if ckpt.get("config_hash") != config.config_hash():
            raise RuntimeError(
                f"Training config changed since last run. Delete {resume_path} or use resume=False."
            )
        model.load_state_dict(ckpt["model"])
        optimizer_g.load_state_dict(ckpt["optimizer_g"])
        scheduler_g.load_state_dict(ckpt["scheduler_g"])
        hist = ckpt["hist"]
        best_val_loss = float(ckpt["best_val_loss"])
        best_epoch = int(ckpt["best_epoch"])
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"Resuming from epoch {start_epoch} (checkpoint {resume_path})", flush=True)
        return start_epoch, hist, best_val_loss, best_epoch

    if history_path.exists():
        prev = torch.load(history_path, map_location="cpu", weights_only=False)
        if prev.get("epoch"):
            last_model = checkpoint_dir / f"last_model_{config.run_name}.pth"
            if last_model.exists():
                model.load_state_dict(torch.load(last_model, map_location=device, weights_only=True))
                hist = prev
                best_val_loss = float(prev.get("best_val_loss", float("inf")))
                best_epoch = int(prev.get("best_epoch", 0))
                start_epoch = int(prev["epoch"][-1]) + 1
                for _ in range(start_epoch - 1):
                    scheduler_g.step()
                print(
                    f"Resuming from epoch {start_epoch} using history + last model (optimizer reset)",
                    flush=True,
                )
                return start_epoch, hist, best_val_loss, best_epoch

    return start_epoch, hist, best_val_loss, best_epoch


def _save_adv_resume_checkpoint(
    resume_path: Path,
    config: AdvNOTrainConfig,
    model: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    scheduler_g: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    hist: dict,
    best_val_loss: float,
    best_epoch: int,
) -> None:
    _atomic_torch_save(
        {
            "config_hash": config.config_hash(),
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "scheduler_g": scheduler_g.state_dict(),
            "hist": hist,
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
        },
        resume_path,
    )


def _forward_batch(model, x, d, h_batch, iD: int):
    return model(x[:, -3 * iD :], d + h_batch / 365.0)


def train_adv_no(
    model: nn.Module,
    data: torch.Tensor,
    mask: torch.Tensor,
    day: torch.Tensor,
    hour: torch.Tensor,
    config: AdvNOTrainConfig,
    *,
    resume: bool = False,
) -> nn.Module:
    device = config.device if torch.cuda.is_available() else "cpu"
    recon_label = "speed L1" if config.loss_metric == "speed" else "uv L1"
    val_recon_label = (
        ("speed RMSE" if config.loss_metric == "speed" else "uv RMSE")
        if config.val_reconstruction_metric == "rmse"
        else recon_label
    )
    print(f"Using device: {device}", flush=True)
    print(f"Loss mode: {config.loss_mode}", flush=True)
    print(f"Channels: iD={config.iD} ({', '.join(config.channel_names[: config.iD])})", flush=True)
    print(f"Reconstruction metric: {config.loss_metric} ({recon_label})", flush=True)
    print(f"Validation metric: {config.val_reconstruction_metric} ({val_recon_label})", flush=True)
    print(f"Optimizer: {config.optimizer} (lr={config.lr:g})", flush=True)
    if config.weight_decay > 0:
        print(f"{config.optimizer} weight_decay: {config.weight_decay:g}", flush=True)
    if config.loss_spatial_mode == "gaussian_center":
        print(
            f"Loss spatial focus: Gaussian center (σ={config.loss_gaussian_sigma} px)",
            flush=True,
        )
    else:
        print("Loss spatial focus: full valid field", flush=True)
    print(f"lambda_perc: {config.lambda_perc} | beta: {config.beta}", flush=True)
    if config.loss_mode == "adv_no":
        perc_label = perceptual_backend_label(config.perceptual_backend, config.lpips_net)
        print(f"perceptual backend: {perc_label}", flush=True)
        if config.perceptual_backend == "vgg":
            print(f"perceptual_weights (VGG layers): {config.perceptual_weights}", flush=True)
        print(
            f"Loss schedule: L1 through epoch {config.perceptual_warmup_epochs}, "
            f"+ perceptual from epoch {config.perceptual_warmup_epochs + 1}, "
            f"+ adversarial from epoch {config.warmup_epochs + 1}",
            flush=True,
        )
        print(
            f"D LR: {'g×' + str(config.d_lr_ratio) if config.d_lr_follow_g else config.d_lr} | "
            f"D skip if loss < {config.d_skip_threshold} | "
            f"G scheduler: {'cosine' if config.use_cosine_lr else 'plateau'}",
            flush=True,
        )

    model = model.to(device)
    discriminator = UNetDiscriminatorSN2D(in_channels=config.iD).to(device)
    perceptual_module = (
        build_perceptual_loss(
            config.perceptual_backend,
            perceptual_weights=config.perceptual_weights,
            lpips_net=config.lpips_net,
        ).to(device)
        if config.loss_mode == "adv_no"
        else None
    )
    adv_loss_fn = AdvNOLoss(
        perceptual_loss=perceptual_module,
        lambda_perc=config.lambda_perc,
        beta=config.beta,
        loss_metric=config.loss_metric,
        val_reconstruction_metric=config.val_reconstruction_metric,
    ).to(device)

    if config.loss_mask_buffer_px > 0:
        invalid_before = float((mask > 0).float().mean())
        mask = expand_invalid_mask_for_loss(mask, config.loss_mask_buffer_px)
        print(
            f"Coastal loss buffer: {config.loss_mask_buffer_px}px land dilation | "
            f"invalid {invalid_before:.2%} -> {float((mask > 0).float().mean()):.2%}",
            flush=True,
        )

    n = len(data)
    val_tensors = None
    if config.val_data_prefix:
        val_tensors = load_training_arrays(config, data_prefix=config.val_data_prefix)
    (
        train_data,
        train_mask,
        train_day,
        train_hour,
        val_data,
        val_mask,
        val_day,
        val_hour,
        train_idx,
        val_idx,
    ) = prepare_train_val_tensors(
        data,
        mask,
        day,
        hour,
        config,
        val_data=val_tensors[0] if val_tensors else None,
        val_mask=val_tensors[1] if val_tensors else None,
        val_day=val_tensors[2] if val_tensors else None,
        val_hour=val_tensors[3] if val_tensors else None,
    )

    train_size = len(train_data)
    val_size = len(val_data)

    train_set = SimulationDataset(train_data, train_mask, train_day, train_hour, iD=config.iD)
    val_set = SimulationDataset(
        val_data, val_mask, val_day, val_hour, mean=train_set.mean, std=train_set.std, iD=config.iD
    )

    run_dir = config.run_dir()
    if not resume and _run_has_artifacts(run_dir, config.run_name):
        _clear_run_artifacts(run_dir, config.run_name)
        print(f"Overwriting previous run artifacts in {run_dir}", flush=True)

    hist = _default_adv_hist()
    best_val_loss = float("inf")
    best_epoch = 0

    config_path, state_path, history_path, resume_path = _adv_paths(config)
    config_path.write_text(json.dumps(asdict(config), indent=2, default=str))

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

    optimizer_g = build_optimizer(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
    )
    initial_d_lr = config.lr * config.d_lr_ratio if config.d_lr_follow_g else config.d_lr
    optimizer_d = build_optimizer(
        discriminator.parameters(),
        lr=initial_d_lr,
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
    )
    if config.use_cosine_lr:
        eta_min = config.cosine_eta_min if config.cosine_eta_min is not None else config.lr * 0.01
        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_g, T_max=config.epochs, eta_min=eta_min
        )
    else:
        scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_g, patience=5, factor=0.5
        )

    if resume:
        start_epoch, hist, best_val_loss, best_epoch = _try_resume_adv_no(
            config,
            model,
            optimizer_g,
            scheduler_g,
            device,
            Path(config.checkpoint_dir),
            history_path,
            resume_path,
        )
    else:
        start_epoch = 1
        if resume_path.exists():
            resume_path.unlink()

    if config.viz_gaussian_circles:
        ensure_circle_l1_hist(
            hist,
            config.viz_gaussian_inner_circle_radii,
            channel_names=config.channel_names,
        )

    norm_stats = {"mean": train_set.mean, "std": train_set.std}
    if start_epoch == 1:
        torch.save(norm_stats, Path(config.checkpoint_dir) / "norm_stats.pt")

    run_info = {
        "run_name": config.run_name,
        "loss_mode": config.loss_mode,
        "data_prefix": config.data_prefix,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "epochs_planned": config.epochs,
        "beta": config.beta,
        "lambda_perc": config.lambda_perc,
        "perceptual_backend": config.perceptual_backend,
        "lpips_net": config.lpips_net,
        "perceptual_warmup_epochs": config.perceptual_warmup_epochs,
        "warmup_epochs": config.warmup_epochs,
        "config_hash": config.config_hash(),
    }
    if start_epoch == 1:
        torch.save(run_info, Path(config.checkpoint_dir) / "run_info.pt")

    train_state = {
        "status": "running",
        "run_name": config.run_name,
        "loss_mode": config.loss_mode,
        "config_hash": config.config_hash(),
        "epochs_planned": config.epochs,
        "last_completed_epoch": 0,
    }
    _save_train_state(state_path, train_state)

    viz_root = Path(config.viz_dir or Path(config.checkpoint_dir) / "viz")
    viz_root.mkdir(parents=True, exist_ok=True)
    curve_perc_warmup = config.perceptual_warmup_epochs if config.loss_mode == "adv_no" else 0
    curve_adv_warmup = config.warmup_epochs if config.loss_mode == "adv_no" else 0
    if config.viz_gaussian_circles:
        (viz_root / "samples_gauss_circles").mkdir(parents=True, exist_ok=True)
    else:
        (viz_root / "samples").mkdir(parents=True, exist_ok=True)
    (viz_root / "curves").mkdir(parents=True, exist_ok=True)
    viz_indices = fixed_viz_indices(config.viz_samples, val_size, config.viz_seed)
    if config.val_data_prefix:
        global_viz_indices = viz_indices
    else:
        global_viz_indices = [int(val_idx[i]) for i in viz_indices]
    split_payload: dict[str, Any] = {
        "mode": "export_split" if config.val_data_prefix else config.val_split_mode,
        "seed": config.val_split_seed,
        "val_split": config.val_split,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
    }
    if config.val_data_prefix:
        split_payload["train_data_prefix"] = config.data_prefix
        split_payload["val_data_prefix"] = config.val_data_prefix
    elif config.merge_val_export_prefix:
        split_payload["train_data_prefix"] = config.data_prefix
        split_payload["merge_val_export_prefix"] = config.merge_val_export_prefix
    (viz_root / "viz_indices.json").write_text(
        json.dumps(
            {
                "seed": config.viz_seed,
                "val_set_indices": viz_indices,
                "export_indices": global_viz_indices,
                "val_split_mode": split_payload["mode"],
                "val_split_seed": config.val_split_seed,
            },
            indent=2,
        )
    )
    (run_dir / "train_val_split.json").write_text(json.dumps(split_payload, indent=2))

    print(f"Checkpoints: {config.checkpoint_dir}", flush=True)
    print(f"History:     {history_path}", flush=True)
    print(
        f"Viz:         {viz_root} (every {config.viz_every} epoch(s), "
        f"fixed val indices {viz_indices})",
        flush=True,
    )
    if resume and start_epoch > 1:
        print(f"Resume:      {resume_path} | continuing from epoch {start_epoch}", flush=True)

    spatial_weight = None  # kept for API compatibility

    viz_paths: list[Path] = []
    curve_paths: list[Path] = []

    crop_h, crop_w = data.shape[-2], data.shape[-1]
    loss_spatial_weight = build_loss_spatial_weight(
        config.loss_spatial_mode, crop_h, crop_w, config.loss_gaussian_sigma
    )
    if loss_spatial_weight is not None:
        stats = gaussian_weight_summary(loss_spatial_weight)
        ring_stats = gaussian_weight_ring_summary(
            loss_spatial_weight, ring_radii=config.viz_gaussian_circle_radii,
        )
        at_radii = {
            f"G(r={r:g})": gaussian_weight_at_radius(loss_spatial_weight, r)
            for r in config.viz_gaussian_circle_radii
        }
        weight_path = Path(config.checkpoint_dir) / "loss_gaussian_weight.pt"
        torch.save(
            {
                "weight": loss_spatial_weight,
                "sigma": config.loss_gaussian_sigma,
                "loss_spatial_mode": config.loss_spatial_mode,
                "ring_radii": config.viz_gaussian_circle_radii,
                "ring_mean_g": ring_stats,
                "g_at_radii": at_radii,
                **stats,
            },
            weight_path,
        )
        print(
            f"Gaussian loss ACTIVE: state⊙G and |pred−target|⊙G | σ={config.loss_gaussian_sigma}px | "
            f"G(center)={stats['center']:.3f} G(edge)={stats['edge']:.3f} "
            f"≈{stats['center_over_edge']:.1f}× at center | saved {weight_path}",
            flush=True,
        )
        radii_note = ", ".join(f"{r:g}px={at_radii[f'G(r={r:g})']:.3f}" for r in config.viz_gaussian_circle_radii)
        print(f"  G on circles: {radii_note}", flush=True)
        ring_note = " | ".join(f"{k} mean G={v:.3f}" for k, v in ring_stats.items())
        print(f"  Ring means: {ring_note}", flush=True)

    if config.preview_loss_mask:
        preview_path = save_loss_mask_preview(
            config.data_prefix,
            viz_root / "loss_mask_preview.png",
            sample_index=0,
            loss_spatial_mode=config.loss_spatial_mode,
            loss_gaussian_sigma=config.loss_gaussian_sigma,
            loss_mask_buffer_px=config.loss_mask_buffer_px,
            channel_names=config.channel_names,
            distance_circle_radii=config.viz_gaussian_circle_radii,
            show=config.show_inline_plots,
        )
        print(f"Loss mask preview: {preview_path}", flush=True)

    if config.preview_val_mask and val_size > 0:
        val_mask_path = save_validation_mask_preview(
            val_set,
            viz_root / "val_mask_preview.png",
            sample_indices=viz_indices,
            global_sample_indices=global_viz_indices,
            channel_names=config.channel_names,
            loss_mask_buffer_px=config.loss_mask_buffer_px,
            show=config.show_inline_plots,
        )
        print(f"Validation mask preview: {val_mask_path}", flush=True)

    if start_epoch > config.epochs:
        print(f"Training already complete ({start_epoch - 1}/{config.epochs} epochs).", flush=True)
        train_state["status"] = "finished"
        train_state["last_completed_epoch"] = config.epochs
        _save_train_state(state_path, train_state)
        return model

    try:
        for epoch in range(start_epoch, config.epochs + 1):
            use_perc = (
                config.loss_mode == "adv_no" and epoch > config.perceptual_warmup_epochs
            )
            use_adv = config.loss_mode == "adv_no" and epoch > config.warmup_epochs

            model.train()
            if use_adv:
                discriminator.train()

            train_total = train_l1 = train_perceptual = train_adv = train_d = 0.0
            d_skips = 0

            for x, y, m, d, h_batch in train_loader:
                x, y, m, d, h_batch = (
                    x.to(device),
                    y.to(device),
                    m.to(device),
                    d.to(device),
                    h_batch.to(device),
                )
                target = y[:, : config.iD]

                optimizer_g.zero_grad()
                pred = _forward_batch(model, x, d, h_batch, config.iD)

                if not use_perc:
                    loss = reconstruction_loss(
                        pred, target, m, metric=config.loss_metric, spatial_weight=loss_spatial_weight
                    )
                    loss.backward()
                    optimizer_g.step()
                    train_total += loss.item() * x.size(0)
                    train_l1 += loss.item() * x.size(0)
                else:
                    g_losses = adv_loss_fn.generator_loss(
                        pred,
                        target,
                        m,
                        discriminator,
                        spatial_weight=loss_spatial_weight,
                        use_perceptual=True,
                        use_adversarial=use_adv,
                    )
                    g_losses["total"].backward()
                    optimizer_g.step()

                    if use_adv:
                        optimizer_d.zero_grad()
                        with torch.no_grad():
                            pred_d = _forward_batch(model, x, d, h_batch, config.iD)
                        d_loss = adv_loss_fn.discriminator_loss(
                            pred_d, target, m, discriminator, spatial_weight=loss_spatial_weight
                        )
                        d_loss_val = d_loss.item()
                        if config.d_skip_threshold <= 0 or d_loss_val >= config.d_skip_threshold:
                            d_loss.backward()
                            optimizer_d.step()
                        else:
                            d_skips += 1
                        train_d += d_loss_val * x.size(0)
                        train_adv += g_losses["adv"].item() * x.size(0)

                    train_total += g_losses["total"].item() * x.size(0)
                    train_l1 += g_losses["l1"].item() * x.size(0)
                    train_perceptual += g_losses["perceptual"].item() * x.size(0)

            train_total /= train_size
            train_l1 /= train_size
            train_perceptual /= train_size if use_perc else 1
            train_adv /= train_size if use_adv else 1
            train_d /= train_size if use_adv else 1

            model.eval()
            val_l1 = val_rmse = val_perceptual = 0.0
            with torch.no_grad():
                for x, y, m, d, h_batch in val_loader:
                    x, y, m, d, h_batch = (
                        x.to(device),
                        y.to(device),
                        m.to(device),
                        d.to(device),
                        h_batch.to(device),
                    )
                    pred = _forward_batch(model, x, d, h_batch, config.iD)
                    target = y[:, : config.iD]
                    metrics = adv_loss_fn.validation_metrics(
                        pred,
                        target,
                        m,
                        spatial_weight=loss_spatial_weight,
                    )
                    val_l1 += metrics["l1"] * x.size(0)
                    val_rmse += metrics["rmse"] * x.size(0)
                    val_perceptual += metrics["perceptual"] * x.size(0)

            val_l1 /= val_size
            val_rmse /= val_size
            val_perceptual /= val_size
            val_loss = val_rmse if config.val_reconstruction_metric == "rmse" else val_l1

            if config.use_cosine_lr:
                scheduler_g.step()
            else:
                scheduler_g.step(val_loss)
            current_lr = optimizer_g.param_groups[0]["lr"]
            if config.d_lr_follow_g:
                current_d_lr = current_lr * config.d_lr_ratio
                for pg in optimizer_d.param_groups:
                    pg["lr"] = current_d_lr
            else:
                current_d_lr = optimizer_d.param_groups[0]["lr"]

            hist["epoch"].append(epoch)
            hist["train"].append(train_total)
            hist["train_l1"].append(train_l1)
            hist["train_perceptual"].append(train_perceptual if use_perc else 0.0)
            hist["train_adv"].append(train_adv if use_adv else 0.0)
            hist["train_d"].append(train_d if use_adv else 0.0)
            hist["val"].append(val_loss)
            hist["val_l1"].append(val_l1)
            hist["val_rmse"].append(val_rmse)
            hist["val_perceptual"].append(val_perceptual)
            hist["lr"].append(current_lr)
            hist["d_lr"].append(current_d_lr)
            hist["best_val_loss"] = best_val_loss
            hist["best_epoch"] = best_epoch
            _atomic_torch_save(hist, history_path)

            if config.viz_gaussian_circles and config.viz_gaussian_inner_circle_radii:
                circle_metrics = compute_viz_sample_circle_l1(
                    model,
                    val_set,
                    device,
                    viz_indices,
                    config.iD,
                    tuple(config.viz_gaussian_inner_circle_radii),
                    channel_names=config.channel_names,
                )
                for radius, ch_metrics in circle_metrics.items():
                    for ch, val in ch_metrics.items():
                        hist[circle_l1_hist_key(ch, radius)].append(val)

            curve_paths = save_adv_no_training_curves(
                hist,
                viz_root,
                perceptual_warmup_epochs=curve_perc_warmup,
                warmup_epochs=curve_adv_warmup,
                loss_metric=config.loss_metric,
                val_reconstruction_metric=config.val_reconstruction_metric,
                perceptual_label=perceptual_backend_label(
                    config.perceptual_backend, config.lpips_net
                ),
            )
            if config.viz_gaussian_circles:
                viz_note = (
                    f"viz idx {viz_indices[0]}"
                    if len(viz_indices) == 1
                    else f"viz mean of {len(viz_indices)} samples"
                )
                circle_curve = save_gaussian_circle_l1_curves(
                    hist,
                    viz_root,
                    config.viz_gaussian_inner_circle_radii,
                    perceptual_warmup_epochs=curve_perc_warmup,
                    warmup_epochs=curve_adv_warmup,
                    viz_sample_note=viz_note,
                )
                if circle_curve is not None:
                    curve_paths.append(circle_curve)

            if val_loss < best_val_loss:
                print(
                    f"epoch {epoch} | val {val_recon_label} {val_loss:.6f} | "
                    f"(improvement of {best_val_loss - val_loss:.6f})",
                    flush=True,
                )
                best_val_loss = val_loss
                best_epoch = epoch
                hist["best_val_loss"] = best_val_loss
                hist["best_epoch"] = best_epoch
                best_model_path = _resolve_best_model_path(config)
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                _atomic_torch_save(hist, history_path)

            torch.save(
                model.state_dict(),
                Path(config.checkpoint_dir) / f"last_model_{config.run_name}.pth",
            )

            if (
                config.checkpoint_every_epochs > 0
                and epoch % config.checkpoint_every_epochs == 0
            ):
                ckpt_dir = Path(config.checkpoint_dir) / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pth")

            train_state["last_completed_epoch"] = epoch
            _save_train_state(state_path, train_state)
            _save_adv_resume_checkpoint(
                resume_path,
                config,
                model,
                optimizer_g,
                scheduler_g,
                epoch,
                hist,
                best_val_loss,
                best_epoch,
            )

            if config.loss_mode != "adv_no":
                mode_tag = "L1-only"
            elif not use_perc:
                mode_tag = "L1-warmup"
            elif not use_adv:
                mode_tag = "L1+perc"
            else:
                mode_tag = "adv-NO"
            d_skip_note = f" | D skips: {d_skips}" if use_adv and d_skips else ""
            lr_note = f"G LR: {current_lr:.2e}"
            if use_adv and config.d_lr_follow_g:
                lr_note += f", D LR: {current_d_lr:.2e}"
            val_detail = (
                f"Val {val_recon_label}: {val_loss:.6f} (L1 {val_l1:.6f}, perc {val_perceptual:.6f})"
                if config.val_reconstruction_metric == "rmse"
                else f"Val {val_recon_label}: {val_l1:.6f} (RMSE {val_rmse:.6f}, perc {val_perceptual:.6f})"
            )
            print(
                f"Epoch {epoch:04d} [{mode_tag}] | "
                f"Train: {train_total:.6f} ({recon_label} {train_l1:.6f}, perc {train_perceptual:.6f}, "
                f"adv {train_adv:.6f}, D {train_d:.6f}) | "
                f"{val_detail} | {lr_note}{d_skip_note}",
                flush=True,
            )

            viz_paths = []
            if config.viz_every > 0 and epoch % config.viz_every == 0:
                epoch_losses = {
                    "val_l1": val_l1,
                    "val_rmse": val_rmse,
                    "train_l1": train_l1,
                    "train": train_total,
                    "train_perceptual": train_perceptual,
                    "train_adv": train_adv,
                    "train_d": train_d,
                }
                pred_cache_dir = (viz_root / "pred_cache") if config.cache_pred_tensors else None
                if config.viz_gaussian_circles:
                    viz_paths = save_gaussian_circles_validation_figures(
                        model,
                        val_set,
                        device,
                        viz_root,
                        iD=config.iD,
                        epoch=epoch,
                        sample_indices=viz_indices,
                        channel_names=config.channel_names,
                        loss_metric=config.loss_metric,
                        epoch_losses=epoch_losses,
                        gaussian_sigma=config.loss_gaussian_sigma,
                        circle_radii=config.viz_gaussian_circle_radii,
                        center_crop_patches=None,
                        inner_circle_radii=config.viz_gaussian_inner_circle_radii,
                        fig_scale=config.viz_gaussian_fig_scale,
                        residual_metric=config.viz_residual_metric,
                        layout="flat_samples",
                        pred_cache_dir=pred_cache_dir,
                    )
                else:
                    viz_paths = save_fix_window_validation_figures(
                        model,
                        val_set,
                        device,
                        viz_root,
                        iD=config.iD,
                        epoch=epoch,
                        sample_indices=viz_indices,
                        residual_metric=config.viz_residual_metric,
                        channel_names=config.channel_names,
                        loss_metric=config.loss_metric,
                        epoch_losses=epoch_losses,
                        layout="flat_samples",
                        pred_cache_dir=pred_cache_dir,
                    )
                if viz_paths:
                    print(
                        f"  saved {len(viz_paths)} sample(s) -> {viz_paths[0].parent}",
                        flush=True,
                    )
            if curve_paths:
                print(f"  updated curves -> {curve_paths[0].parent}", flush=True)
            if config.show_inline_plots and viz_paths:
                display_adv_no_epoch_plots(viz_paths, curve_paths=[], show_curves=False)

    except KeyboardInterrupt:
        print(
            "\nTraining stopped by user. Progress saved through last completed epoch.",
            flush=True,
        )
        train_state["status"] = "stopped"
        _save_train_state(state_path, train_state)
        return model

    run_info["finished_at"] = datetime.now().isoformat(timespec="seconds")
    run_info["best_val_loss"] = best_val_loss
    run_info["best_epoch"] = best_epoch
    torch.save(run_info, Path(config.checkpoint_dir) / "run_info.pt")

    train_state["status"] = "finished"
    train_state["best_val_loss"] = best_val_loss
    train_state["best_epoch"] = best_epoch
    _save_train_state(state_path, train_state)

    print(
        f"\nTraining complete. Best val {val_recon_label}: {best_val_loss:.6f} (epoch {best_epoch})",
        flush=True,
    )
    return model


def run_adv_no_training(config: AdvNOTrainConfig, *, resume: bool = False) -> nn.Module:
    if config.merge_val_export_prefix and not config.val_data_prefix:
        data, mask, day, hour, crop_h, crop_w = load_merged_training_arrays(config)
        print(
            f"Dataset: {config.data_prefix} + {config.merge_val_export_prefix} "
            f"(merged, {config.val_split_mode} split, seed={config.val_split_seed})",
            flush=True,
        )
    else:
        data, mask, day, hour, crop_h, crop_w = load_training_arrays(config)
        print(f"Dataset: {config.data_prefix}", flush=True)
    print(f"Samples: {len(data)} | crop {crop_h} x {crop_w}", flush=True)
    model = build_model(crop_h, crop_w, config)
    return train_adv_no(model, data, mask, day, hour, config, resume=resume)
