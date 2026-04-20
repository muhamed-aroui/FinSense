#!/usr/bin/env python3
"""Entry point for Phase 2 training runs.

Dispatches to the encoder fine-tuning or LoRA trainer based on the
config's ``method`` field.

Usage:
    uv run scripts/train_phase2.py --config configs/phase2-deberta-ft-sampler.yaml
    uv run scripts/train_phase2.py --config configs/phase2-llama-lora.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


TRACKIO_PROJECT = "finsense"


def _log_to_trackio(results: dict) -> None:
    """Initialize a Trackio run, log curves + summary metrics, and finish."""
    import trackio

    run_config = {
        "run_id": results["run_id"],
        "config_hash": results["config_hash"],
        "git_sha": results["git_sha"],
        "phase": results["phase"],
        "model": results["model"],
        "method": results.get("method", "unknown"),
        "head": results["head"],
        "imbalance_strategy": results["imbalance_strategy"],
    }

    trackio.init(
        project=TRACKIO_PROJECT,
        name=results["run_id"],
        group=f"phase_{results['phase']}",
        config=run_config,
        space_id="umoeria/finsense",
    )

    try:
        # Per-epoch curves
        train_curve = results.get("train_loss_curve") or []
        val_curve = results.get("val_macro_f1_curve") or []
        for epoch, (tl, vf1) in enumerate(zip(train_curve, val_curve)):
            trackio.log({"train_loss": tl, "val_macro_f1": vf1}, step=epoch)

        # Final summary scalars
        summary = {
            "macro_f1": results["macro_f1"],
            "inference_latency_ms": results["inference_latency_ms"],
            "trainable_params": results["trainable_params"],
            "peak_vram_gb": results["peak_vram_gb"],
            "wall_clock_min": results["wall_clock_min"],
            "best_epoch": results["best_epoch"],
        }
        for i, f1 in enumerate(results.get("per_class_f1", [])):
            summary[f"per_class_f1/{i}"] = f1

        # LoRA-specific fields
        if "lora_trainable_params" in results:
            summary["lora_trainable_params"] = results["lora_trainable_params"]
            summary["lora_r"] = results["lora_r"]

        trackio.log(summary)
    finally:
        trackio.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="FinSense Phase 2 training")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--no-trackio", action="store_true",
        help="Skip Trackio logging (for local debugging only)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    from finsense.training.trainer import load_config

    cfg = load_config(config_path)
    method = cfg.get("method", "full_ft")

    print(f"Starting run: {cfg['run_id']}")
    print(f"  Model: {cfg['model']}")
    print(f"  Method: {method}")
    print(f"  Head: {cfg['head']}")
    print(f"  Imbalance: {cfg['imbalance_strategy']}")
    print()

    if method == "full_ft":
        from finsense.training.fine_tune import train_encoder_ft
        results = train_encoder_ft(config_path)
    elif method in ("lora", "qlora"):
        from finsense.training.lora_trainer import train_lora
        results = train_lora(config_path)
    else:
        print(f"Error: unknown method {method!r}", file=sys.stderr)
        sys.exit(1)

    if not args.no_trackio:
        try:
            _log_to_trackio(results)
            print("\nTrackio: metrics logged successfully.")
        except Exception as e:
            print(f"\nWarning: Trackio logging failed: {e}", file=sys.stderr)
            print("Results are saved to artifacts/ as JSON.", file=sys.stderr)

    print(f"\nDone. Results saved to {cfg['output_dir']}/results.json")


if __name__ == "__main__":
    main()
