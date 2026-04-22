"""Full fine-tuning training loop for encoder models (DeBERTa, FinBERT).

Unlike Phase 1's frozen-backbone trainer, this unfreezes the entire model
and trains with a lower learning rate. Uses AutoModelForSequenceClassification
with a built-in linear head.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from finsense.data.dataset import load_splits
from finsense.evaluation.metrics import compute_metrics, measure_latency_model
from finsense.training.losses import build_loss, compute_class_weights
from finsense.training.sampler import build_weighted_sampler
from finsense.training.trainer import (
    _get_device,
    _get_dtype,
    config_hash,
    git_sha,
    load_config,
    setup_determinism,
)


def train_encoder_ft(config_path: str | Path) -> dict[str, Any]:
    """Run a full fine-tuning training run for an encoder model.

    Returns a results dict with all Trackio-logged fields.
    """
    config_path = Path(config_path)
    cfg = load_config(config_path)
    cfg_hash = config_hash(config_path)
    sha = git_sha()

    # --- Determinism ---
    setup_determinism(cfg["seed"])

    # --- Device & precision ---
    device = _get_device(cfg)
    dtype = _get_dtype(cfg)
    use_amp = dtype in (torch.float16, torch.bfloat16) and device.type == "cuda"
    # GradScaler is only needed for FP16. BF16 has FP32-range exponents and
    # does not suffer from gradient underflow, so scaling is a no-op at best.
    use_scaler = dtype == torch.float16 and device.type == "cuda"

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
    # "combo" means sampler + weighted_ce: use sampler for data, weighted_ce for loss
    use_sampler = imbalance in ("sampler", "combo")
    sampler = build_weighted_sampler(train_ds.labels) if use_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=0, pin_memory=device.type == "cuda")

    # --- Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], num_labels=3, torch_dtype=torch.float32,   # AMP requires FP32 master weights
    ).to(device)
    assert next(model.parameters()).dtype == torch.float32, (
    f"Model params must be FP32 for AMP, got {next(model.parameters()).dtype}"
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- Loss ---
    # For combo: use weighted_ce loss
    # For sampler-only: use plain CE
    loss_strategy = "weighted_ce" if imbalance == "combo" else imbalance
    class_weights = None
    if loss_strategy in ("weighted_ce", "focal"):
        cw_cfg = cfg.get("class_weights", "balanced")
        if cw_cfg == "balanced":
            class_weights = compute_class_weights(train_ds.labels)
        else:
            class_weights = list(cw_cfg)

    loss_fn = build_loss(
        strategy=loss_strategy,
        class_weights=class_weights,
        focal_gamma=cfg.get("focal_gamma", 2.0),
    ).to(device)

    # --- Optimizer & scheduler ---
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    total_steps = cfg["max_epochs"] * len(train_loader)
    warmup_steps = int(cfg.get("warmup_ratio", 0.1) * total_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))

    # --- Output dir ---
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    #scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    patience = cfg.get("early_stopping_patience", 2)
    max_wall = cfg.get("max_wall_clock_min", 30) * 60
    global_step = 0

    train_loss_curve: list[float] = []
    val_f1_curve: list[float] = []

    start_time = time.time()

    for epoch in range(cfg["max_epochs"]):
        elapsed = time.time() - start_time
        if elapsed > max_wall:
            print(f"Wall-clock budget exceeded ({elapsed/60:.1f} min). Stopping.")
            break

        # --- Train epoch ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['max_epochs']}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)

            if not torch.isfinite(loss):
              raise RuntimeError(f"Non-finite loss at step {global_step}: {loss.item()}. "
                       f"Check precision (DeBERTa-v3 is unstable in bf16) and gradient clipping.")
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                          # unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Linear warmup then cosine
            if global_step < warmup_steps:
                lr_scale = (global_step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = cfg["learning_rate"] * lr_scale
            else:
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_loss_curve.append(avg_loss)

        # --- Val epoch ---
        val_metrics = _evaluate_model(model, val_loader, device, dtype, use_amp)
        val_f1 = val_metrics["macro_f1"]
        val_f1_curve.append(val_f1)

        print(
            f"  Epoch {epoch+1}: train_loss={avg_loss:.4f}  "
            f"val_macro_f1={val_f1:.4f}  "
            f"val_per_class={val_metrics['per_class_f1']}"
        )

        # --- Checkpoint every epoch ---
        torch.save(
            {"epoch": epoch + 1, "model_state_dict": model.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "val_macro_f1": val_f1, "config": cfg},
            output_dir / f"epoch-{epoch+1}.pt",
        )

        # --- Early stopping ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "val_macro_f1": val_f1, "config": cfg},
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
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = _evaluate_model(model, test_loader, device, dtype, use_amp)
    latency_ms = measure_latency_model(model, test_ds, device, dtype)

    peak_vram_gb = 0.0
    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / 1e9

    results = {
        "run_id": cfg["run_id"],
        "config_hash": cfg_hash,
        "git_sha": sha,
        "phase": cfg["phase"],
        "model": cfg["model"],
        "method": cfg.get("method", "full_ft"),
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


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    use_amp: bool,
) -> dict[str, Any]:
    """Run evaluation with a single model (no separate backbone/head)."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    model.train()
    return compute_metrics(all_labels, all_preds)
