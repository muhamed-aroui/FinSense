#!/usr/bin/env python3
"""Entry point for training a single Phase 1 head-sweep run.

Usage:
    uv run scripts/train.py --config configs/phase1-linear-baseline.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="FinSense Phase 1 training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config file in configs/",
    )
    parser.add_argument(
        "--no-trackio",
        action="store_true",
        help="Skip Trackio logging (for local debugging only)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    from finsense.training.trainer import load_config, train

    cfg = load_config(config_path)
    print(f"Starting run: {cfg['run_id']}")
    print(f"  Head: {cfg['head']}")
    print(f"  Imbalance: {cfg['imbalance_strategy']}")
    print(f"  Model: {cfg['model']}")
    print()

    results = train(config_path)

    # Trackio logging
    if not args.no_trackio:
        try:
            import trackio

            trackio.log(
                run_id=results["run_id"],
                config_hash=results["config_hash"],
                git_sha=results["git_sha"],
                phase=results["phase"],
                model=results["model"],
                head=results["head"],
                imbalance_strategy=results["imbalance_strategy"],
                macro_f1=results["macro_f1"],
                per_class_f1=results["per_class_f1"],
                inference_latency_ms=results["inference_latency_ms"],
                trainable_params=results["trainable_params"],
                train_loss_curve=results["train_loss_curve"],
                val_macro_f1_curve=results["val_macro_f1_curve"],
                peak_vram_gb=results["peak_vram_gb"],
                wall_clock_min=results["wall_clock_min"],
                best_epoch=results["best_epoch"],
            )
            print("\nTrackio: metrics logged successfully.")
        except Exception as e:
            print(f"\nWarning: Trackio logging failed: {e}", file=sys.stderr)
            print("Results are saved to artifacts/ as JSON.", file=sys.stderr)

    print(f"\nDone. Results saved to {cfg['output_dir']}/results.json")


if __name__ == "__main__":
    main()
