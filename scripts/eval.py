#!/usr/bin/env python3
"""Evaluate a trained Phase 1 checkpoint on the test set.

Usage:
    uv run scripts/eval.py --config configs/phase1-linear-baseline.yaml \
                           --checkpoint artifacts/phase1-linear-baseline/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="FinSense Phase 1 evaluation")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    args = parser.parse_args()

    config_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)

    if not config_path.exists():
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    from finsense.data.dataset import load_splits
    from finsense.evaluation.metrics import compute_metrics, measure_latency
    from finsense.models.backbone import FrozenBackbone
    from finsense.models.heads import build_head
    from finsense.training.trainer import load_config

    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Load model
    backbone = FrozenBackbone(cfg["model"]).to(device)
    head = build_head(cfg["head"], backbone.hidden_size, cfg.get("head_config")).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    head.load_state_dict(ckpt["head_state_dict"])

    # Load test data
    splits = load_splits(model_name=cfg["model"], max_length=cfg["max_length"])
    test_ds = splits["test"]

    from torch.utils.data import DataLoader

    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    # Evaluate
    head.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    use_amp = dtype != torch.float32 and device.type == "cuda"

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
                hidden = backbone(input_ids, attention_mask)
                logits = head(hidden, attention_mask)

            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    metrics = compute_metrics(all_labels, all_preds)
    latency = measure_latency(backbone, head, test_ds, device, dtype)

    # Print results
    print(f"Run: {cfg['run_id']}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Macro-F1:        {metrics['macro_f1']:.4f}")
    print(f"Per-class F1:    {metrics['per_class_f1']}")
    print(f"Balanced Acc:    {metrics['balanced_accuracy']:.4f}")
    print(f"Latency:         {latency:.2f} ms/example")
    print(f"Confusion matrix:")
    for row in metrics["confusion_matrix"]:
        print(f"  {row}")

    # Save
    output = {**metrics, "inference_latency_ms": latency, "run_id": cfg["run_id"]}
    out_path = Path(cfg["output_dir"]) / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
