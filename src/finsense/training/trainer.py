"""Training loop for the Phase 1 frozen-backbone head sweep.

Handles: config loading, determinism setup, training with early stopping,
per-epoch checkpointing, and Trackio metric logging.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from finsense.data.dataset import SentimentDataset, load_splits
from finsense.evaluation.metrics import compute_metrics, measure_latency
from finsense.models.backbone import FrozenBackbone
from finsense.models.heads import build_head
from finsense.training.losses import build_loss, compute_class_weights
from finsense.training.sampler import build_weighted_sampler


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return the dict."""
    config_path = Path(config_path)
    with config_path.open() as f:
        return yaml.safe_load(f)


def config_hash(config_path: str | Path) -> str:
    """SHA-256 hex digest of the config file contents."""
    return hashlib.sha256(Path(config_path).read_bytes()).hexdigest()


def git_sha() -> str:
    """Return the current git HEAD SHA, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def setup_determinism(seed: int) -> None:
    """Set all seeds and enable deterministic algorithms per CLAUDE.md §4.1."""
    import random

    import numpy as np
    from transformers import set_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _get_device(cfg: dict) -> torch.device:
    """Pick device: CUDA if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_dtype(cfg: dict) -> torch.dtype:
    """Pick precision dtype from config. Fallback fp16 → fp32 on CPU."""
    precision = cfg.get("precision", "bf16")
    device = _get_device(cfg)
    if device.type == "cpu":
        return torch.float32
    if precision == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16  # fallback
    if precision == "fp16":
        return torch.float16
    return torch.float32


def train(config_path: str | Path) -> dict[str, Any]:
    """Run a single training run defined by a YAML config.

    Returns a results dict with all Trackio-logged fields.
    """
    config_path = Path(config_path)
    cfg = load_config(config_path)
    cfg_hash = config_hash(config_path)
    sha = git_sha()

    # --- Determinism ---
    seed = cfg["seed"]
    setup_determinism(seed)

    # --- Device & precision ---
    device = _get_device(cfg)
    dtype = _get_dtype(cfg)
    use_amp = dtype in (torch.float16, torch.bfloat16) and device.type == "cuda"

    # --- Data ---
    splits = load_splits(
        model_name=cfg["model"],
        max_length=cfg["max_length"],
    )
    train_ds = splits["train"]
    val_ds = splits["val"]
    test_ds = splits["test"]

    # --- Sampler ---
    imbalance = cfg["imbalance_strategy"]
    sampler = None
    shuffle = True
    if imbalance == "sampler":
        sampler = build_weighted_sampler(train_ds.labels)
        shuffle = False  # sampler handles ordering

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # --- Model ---
    backbone = FrozenBackbone(cfg["model"]).to(device)
    head = build_head(
        cfg["head"],
        backbone.hidden_size,
        cfg.get("head_config"),
    ).to(device)

    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)

    # --- Loss ---
    class_weights = None
    if imbalance in ("weighted_ce", "focal"):
        cw_cfg = cfg.get("class_weights", "balanced")
        if cw_cfg == "balanced":
            class_weights = compute_class_weights(train_ds.labels)
        else:
            class_weights = list(cw_cfg)

    loss_fn = build_loss(
        strategy=imbalance,
        class_weights=class_weights,
        focal_gamma=cfg.get("focal_gamma", 2.0),
    ).to(device)

    # --- Optimizer & scheduler ---
    optimizer = AdamW(
        head.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    total_steps = cfg["max_epochs"] * len(train_loader)
    warmup_steps = int(cfg.get("warmup_ratio", 0.1) * total_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    # --- Output dir ---
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    patience = cfg.get("early_stopping_patience", 3)
    max_wall = cfg.get("max_wall_clock_min", 20) * 60  # seconds

    train_loss_curve: list[float] = []
    val_f1_curve: list[float] = []

    start_time = time.time()

    for epoch in range(cfg["max_epochs"]):
        # Check wall-clock budget
        elapsed = time.time() - start_time
        if elapsed > max_wall:
            print(f"Wall-clock budget exceeded ({elapsed/60:.1f} min). Stopping.")
            break

        # --- Train epoch ---
        head.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['max_epochs']}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                hidden = backbone(input_ids, attention_mask)
                # Detach hidden states — backbone is frozen, but making
                # the boundary explicit avoids any accidental graph leaks.
                hidden = hidden.detach()
                logits = head(hidden, attention_mask)
                loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Linear warmup then cosine
            if n_batches < warmup_steps:
                lr_scale = (n_batches + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg["learning_rate"] * lr_scale
            else:
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_loss_curve.append(avg_loss)

        # --- Val epoch ---
        val_metrics = _evaluate(backbone, head, val_loader, device, dtype, use_amp)
        val_f1 = val_metrics["macro_f1"]
        val_f1_curve.append(val_f1)

        print(
            f"  Epoch {epoch+1}: train_loss={avg_loss:.4f}  "
            f"val_macro_f1={val_f1:.4f}  "
            f"val_per_class={val_metrics['per_class_f1']}"
        )

        # --- Checkpoint every epoch (CLAUDE.md §4.4) ---
        ckpt_path = output_dir / f"epoch-{epoch+1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "head_state_dict": head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "config": cfg,
            },
            ckpt_path,
        )

        # --- Early stopping ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best checkpoint
            torch.save(
                {
                    "epoch": epoch + 1,
                    "head_state_dict": head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": val_f1,
                    "config": cfg,
                },
                output_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    wall_clock_min = (time.time() - start_time) / 60.0

    # --- Load best checkpoint and evaluate on test ---
    best_ckpt = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    head.load_state_dict(best_ckpt["head_state_dict"])

    test_metrics = _evaluate(backbone, head, test_loader, device, dtype, use_amp)
    latency_ms = measure_latency(backbone, head, test_ds, device, dtype)

    # Peak VRAM
    peak_vram_gb = 0.0
    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9

    results = {
        "run_id": cfg["run_id"],
        "config_hash": cfg_hash,
        "git_sha": sha,
        "phase": cfg["phase"],
        "model": cfg["model"],
        "head": cfg["head"],
        "imbalance_strategy": imbalance,
        "macro_f1": test_metrics["macro_f1"],
        "per_class_f1": test_metrics["per_class_f1"],
        "inference_latency_ms": latency_ms,
        "trainable_params": trainable_params,
        "train_loss_curve": train_loss_curve,
        "val_macro_f1_curve": val_f1_curve,
        "peak_vram_gb": peak_vram_gb,
        "wall_clock_min": wall_clock_min,
        "best_epoch": best_epoch,
    }

    # Save results as JSON alongside checkpoints
    import json
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n{'='*60}")
    print(f"Run: {cfg['run_id']}")
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Per-class F1: {test_metrics['per_class_f1']}")
    print(f"Latency: {latency_ms:.2f} ms/example")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"Wall-clock: {wall_clock_min:.1f} min")
    print(f"Best epoch: {best_epoch}")
    print(f"{'='*60}")

    return results


def _evaluate(
    backbone: FrozenBackbone,
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    use_amp: bool,
) -> dict[str, Any]:
    """Run evaluation on a data loader, return metrics dict."""
    head.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                hidden = backbone(input_ids, attention_mask)
                logits = head(hidden, attention_mask)

            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    head.train()
    return compute_metrics(all_labels, all_preds)
