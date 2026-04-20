"""LoRA / QLoRA training loop for Llama-based classification.

Handles: PEFT adapter setup, 4-bit quantization, gradient accumulation,
last-token pooling classification, early stopping, and Trackio logging.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from finsense.data.dataset import load_splits
from finsense.evaluation.metrics import compute_metrics, measure_latency_model
from finsense.models.llama_classifier import apply_lora, build_llama_classifier
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


def train_lora(config_path: str | Path) -> dict[str, Any]:
    """Run a LoRA/QLoRA training run for Llama classification.

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
    model = build_llama_classifier(cfg)
    model = apply_lora(model, cfg)

    # Move classifier head to device (backbone may already be on device via device_map)
    if not hasattr(model.backbone, "hf_device_map"):
        model = model.to(device)
    else:
        model.classifier = model.classifier.to(device)

    # Count trainable params (LoRA + classifier head)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(
        p.numel() for n, p in model.backbone.named_parameters() if p.requires_grad
    )

    # --- Loss ---
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
    ).to(device)

    # --- Optimizer & scheduler ---
    # Only optimize trainable parameters (LoRA adapters + classifier)
    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_param_list,
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    grad_accum_steps = cfg.get("gradient_accumulation_steps", 1)
    steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = cfg["max_epochs"] * steps_per_epoch
    warmup_steps = int(cfg.get("warmup_ratio", 0.1) * total_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))

    # --- Output dir ---
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and "quantization" not in cfg)
    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    patience = cfg.get("early_stopping_patience", 2)
    max_wall = cfg.get("max_wall_clock_min", 45) * 60
    global_step = 0

    train_loss_curve: list[float] = []
    val_f1_curve: list[float] = []

    start_time = time.time()

    for epoch in range(cfg["max_epochs"]):
        elapsed = time.time() - start_time
        if elapsed > max_wall:
            print(f"Wall-clock budget exceeded ({elapsed/60:.1f} min). Stopping.")
            break

        model.train()
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{cfg['max_epochs']}", leave=False
        )):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if global_step < warmup_steps:
                    lr_scale = (global_step + 1) / warmup_steps
                    for pg in optimizer.param_groups:
                        pg["lr"] = cfg["learning_rate"] * lr_scale
                else:
                    scheduler.step()

                global_step += 1

            epoch_loss += loss.item() * grad_accum_steps
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_loss_curve.append(avg_loss)

        # --- Val epoch ---
        val_metrics = _evaluate_llama(model, val_loader, device, dtype, use_amp)
        val_f1 = val_metrics["macro_f1"]
        val_f1_curve.append(val_f1)

        print(
            f"  Epoch {epoch+1}: train_loss={avg_loss:.4f}  "
            f"val_macro_f1={val_f1:.4f}  "
            f"val_per_class={val_metrics['per_class_f1']}"
        )

        # --- Checkpoint every epoch ---
        ckpt_data = {
            "epoch": epoch + 1,
            "classifier_state_dict": model.classifier.state_dict(),
            "val_macro_f1": val_f1,
            "config": cfg,
        }
        torch.save(ckpt_data, output_dir / f"epoch-{epoch+1}.pt")
        # Save LoRA adapters separately
        model.backbone.save_pretrained(output_dir / f"lora-epoch-{epoch+1}")

        # --- Early stopping ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(ckpt_data, output_dir / "best.pt")
            model.backbone.save_pretrained(output_dir / "lora-best")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    wall_clock_min = (time.time() - start_time) / 60.0

    # --- Load best and evaluate on test ---
    best_ckpt = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.classifier.load_state_dict(best_ckpt["classifier_state_dict"])
    # Reload best LoRA adapters
    from peft import PeftModel
    model.backbone.load_adapter(str(output_dir / "lora-best"), adapter_name="default")

    test_metrics = _evaluate_llama(model, test_loader, device, dtype, use_amp)
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
        "method": cfg.get("method", "lora"),
        "head": cfg["head"],
        "imbalance_strategy": imbalance,
        "macro_f1": test_metrics["macro_f1"],
        "per_class_f1": test_metrics["per_class_f1"],
        "inference_latency_ms": latency_ms,
        "trainable_params": trainable_params,
        "lora_trainable_params": lora_params,
        "train_loss_curve": train_loss_curve,
        "val_macro_f1_curve": val_f1_curve,
        "peak_vram_gb": peak_vram_gb,
        "wall_clock_min": wall_clock_min,
        "best_epoch": best_epoch,
        "lora_r": cfg["lora_config"]["r"],
        "base_model_precision": "4bit" if "quantization" in cfg else "fp16",
        "gradient_accumulation_steps": grad_accum_steps,
    }

    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n{'='*60}")
    print(f"Run: {cfg['run_id']}")
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Per-class F1: {test_metrics['per_class_f1']}")
    print(f"Latency: {latency_ms:.2f} ms/example")
    print(f"Trainable params: {trainable_params:,} (LoRA: {lora_params:,})")
    print(f"Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"Wall-clock: {wall_clock_min:.1f} min")
    print(f"Best epoch: {best_epoch}")
    print(f"{'='*60}")

    return results


def _evaluate_llama(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    use_amp: bool,
) -> dict[str, Any]:
    """Evaluate a LlamaClassifier on a data loader."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                logits = model(input_ids, attention_mask)

            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    model.train()
    return compute_metrics(all_labels, all_preds)
