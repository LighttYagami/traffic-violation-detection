"""
train.py
=========
Main training script for YOLO26s fine-tuning.
Loads config from YAML, registers custom callbacks, runs training,
and auto-evaluates on test split after training.

Usage:
    python train.py --config ../config/train_config.yaml

    # Override experiment name:
    python train.py --config ../config/train_config.yaml --name exp_002_imgsz640

    tmux new -s lm
    conda activate lmcf
    CUDA_VISIBLE_DEVICES=0 python train_v2.py

    Detach: Ctrl+B then D — job keeps running
    Re-attach later: tmux attach -t lm
    conda deactivate
"""

import os
import sys
import yaml
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path for callback imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ultralytics import YOLO
from callbacks.training_logger import TrainingLogger


def load_config(config_path: str) -> dict:
    """Load training config from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment(config: dict, base_dir: str = None) -> str:
    """
    Create experiment directory and save config snapshot.
    Returns path to experiment directory.
    """
    exp_name = config["experiment_name"]
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "..", "training_logs")

    exp_dir = os.path.join(base_dir, exp_name)

    # Don't overwrite existing experiments
    if os.path.exists(exp_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = f"{exp_dir}_{timestamp}"
        print(f"Experiment directory exists. Using: {exp_dir}")

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)

    # Save config snapshot
    snapshot_path = os.path.join(exp_dir, "config_snapshot.yaml")
    with open(snapshot_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config snapshot saved: {snapshot_path}")

    return exp_dir


def train(config_path: str, name_override: str = None):
    """Main training function."""

    # ── Load config ──
    config = load_config(config_path)
    if name_override:
        config["experiment_name"] = name_override

    print("=" * 60)
    print(f"  Experiment: {config['experiment_name']}")
    print(f"  Model:      {config['model']['architecture']}")
    print(f"  Image size: {config['model']['input_size']}")
    print(f"  Epochs:     {config['training']['epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print("=" * 60)

    # ── Setup experiment directory ──
    exp_dir = setup_experiment(config)

    # ── Initialize model ──
    model_arch = config["model"]["architecture"]
    print(f"\nLoading pretrained model: {model_arch}")
    model = YOLO(model_arch)

    # ── Register custom callbacks ──
    logger = TrainingLogger(save_dir=exp_dir, config=config)
    logger.register(model)
    print("Custom training logger registered")

    # ── Resolve data.yaml path ──
    data_yaml = config["data"]["yaml_path"]
    if not os.path.isabs(data_yaml):
        # Relative to config file location
        config_dir = os.path.dirname(os.path.abspath(config_path))
        data_yaml = os.path.normpath(os.path.join(config_dir, data_yaml))
    print(f"Dataset config: {data_yaml}")

    # Verify data.yaml exists
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    # ── Training arguments ──
    train_cfg = config["training"]
    aug_cfg = config["augmentation"]
    ckpt_cfg = config["checkpointing"]
    lr_cfg = train_cfg["learning_rate"]

    # ── Start training ──
    print("\nStarting training...\n")
    results = model.train(
        # Data
        data=data_yaml,

        # Training
        epochs=train_cfg["epochs"],
        batch=train_cfg["batch_size"],
        imgsz=config["model"]["input_size"],
        device=0,
        patience=train_cfg["patience"],
        amp=train_cfg["amp"],
        freeze=train_cfg["freeze_layers"],

        # Learning rate
        lr0=lr_cfg["initial"],
        lrf=lr_cfg["final_factor"],
        warmup_epochs=lr_cfg["warmup_epochs"],
        warmup_momentum=lr_cfg["warmup_momentum"],
        warmup_bias_lr=lr_cfg["warmup_bias_lr"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],

        # Augmentation
        hsv_h=aug_cfg["hsv_h"],
        hsv_s=aug_cfg["hsv_s"],
        hsv_v=aug_cfg["hsv_v"],
        degrees=aug_cfg["degrees"],
        translate=aug_cfg["translate"],
        scale=aug_cfg["scale"],
        shear=aug_cfg["shear"],
        perspective=aug_cfg["perspective"],
        flipud=aug_cfg["flipud"],
        fliplr=aug_cfg["fliplr"],
        mosaic=aug_cfg["mosaic"],
        mixup=aug_cfg["mixup"],
        copy_paste=aug_cfg["copy_paste"],

        # Checkpointing
        save_period=ckpt_cfg["save_period"],
        save=True,

        # Output
        project=exp_dir,
        name="yolo_output",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    # ── Copy best.pt to experiment weights directory ──
    yolo_output_dir = os.path.join(exp_dir, "yolo_output")
    best_src = os.path.join(yolo_output_dir, "weights", "best.pt")
    last_src = os.path.join(yolo_output_dir, "weights", "last.pt")

    if os.path.exists(best_src):
        shutil.copy2(best_src, os.path.join(exp_dir, "weights", "best.pt"))
        print(f"\nbest.pt copied to {exp_dir}/weights/")
    if os.path.exists(last_src):
        shutil.copy2(last_src, os.path.join(exp_dir, "weights", "last.pt"))

    # ── Copy Ultralytics results.csv ──
    results_csv = os.path.join(yolo_output_dir, "results.csv")
    if os.path.exists(results_csv):
        shutil.copy2(results_csv, os.path.join(exp_dir, "metrics", "results.csv"))

    # ── Copy auto-generated plots ──
    for plot_file in Path(yolo_output_dir).glob("*.png"):
        shutil.copy2(str(plot_file), os.path.join(exp_dir, "plots", plot_file.name))
    for plot_file in Path(yolo_output_dir).glob("*.jpg"):
        shutil.copy2(str(plot_file), os.path.join(exp_dir, "plots", plot_file.name))

    # ── Run test set evaluation ──
    print("\n" + "=" * 60)
    print("  Running evaluation on test split...")
    print("=" * 60)

    best_model = YOLO(os.path.join(exp_dir, "weights", "best.pt"))
    test_metrics = best_model.val(
        data=data_yaml,
        split="test",
        plots=True,
        conf=0.25,
        iou=0.5,
        verbose=True,
        project=exp_dir,
        name="test_eval",
        exist_ok=True,
    )

    # Save test metrics
    test_report = {
        "mAP50": float(test_metrics.box.map50),
        "mAP50_95": float(test_metrics.box.map),
        "precision": float(test_metrics.box.mp),
        "recall": float(test_metrics.box.mr),
    }

    # Per-class metrics
    if hasattr(best_model, 'names'):
        test_report["per_class"] = {}
        for i, name in best_model.names.items():
            try:
                test_report["per_class"][name] = {
                    "ap50": float(test_metrics.box.ap50[i]),
                    "ap50_95": float(test_metrics.box.maps[i]),
                }
            except (IndexError, AttributeError):
                pass

    import json
    test_report_path = os.path.join(exp_dir, "metrics", "test_results.json")
    with open(test_report_path, "w") as f:
        json.dump(test_report, f, indent=2)

    print(f"\nTest results saved: {test_report_path}")
    print(f"\n  Test mAP@50:    {test_report['mAP50']:.4f}")
    print(f"  Test mAP@50-95: {test_report['mAP50_95']:.4f}")
    print(f"  Test Precision: {test_report['precision']:.4f}")
    print(f"  Test Recall:    {test_report['recall']:.4f}")

    print(f"\n  All artifacts saved to: {exp_dir}")
    print(f"  Best model:     {exp_dir}/weights/best.pt")
    print(f"  Training logs:  {exp_dir}/metrics/custom_metrics.json")
    print(f"  Results CSV:    {exp_dir}/metrics/results.csv")
    print(f"  Plots:          {exp_dir}/plots/")

    return exp_dir


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26s for traffic violation detection")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "config", "train_config.yaml"),
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Override experiment name from config",
    )
    args = parser.parse_args()
    train(args.config, args.name)


if __name__ == "__main__":
    main()
