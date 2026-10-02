"""
training_logger.py
===================
Custom Ultralytics callback that logs training dynamics beyond
what results.csv provides: gradient norms, weight norms, per-class AP.

All data is saved to custom_metrics.json with atomic writes (crash-safe).

Usage:
    from callbacks.training_logger import TrainingLogger

    logger = TrainingLogger(save_dir="training_logs/exp_001", config=config_dict)

    model.add_callback("on_train_epoch_end", logger.on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", logger.on_fit_epoch_end)
    model.add_callback("on_model_save", logger.on_model_save)
    model.add_callback("on_train_end", logger.on_train_end)
"""

import os
import json
import time
import shutil
import torch
from collections import defaultdict


# ─── YOLO26s layer grouping ─────────────────────────────────────────────────
# Determined by inspecting model.named_parameters()
# Adjust if architecture changes
LAYER_GROUPS = {
    "backbone": list(range(0, 10)),   # model.model.0 - model.model.9
    "neck": list(range(10, 23)),      # model.model.10 - model.model.21
    "head": list(range(23, 24)),      # model.model.22+
}


def _get_layer_group(param_name: str) -> str:
    """Determine which group a parameter belongs to based on its name."""
    # param names look like: model.model.0.conv.weight, model.model.15.m.0.weight
    try:
        parts = param_name.split(".")
        if len(parts) >= 3 and parts[1] == "model":
            layer_idx = int(parts[2])
            for group_name, indices in LAYER_GROUPS.items():
                if layer_idx in indices:
                    return group_name
    except (ValueError, IndexError):
        pass
    return "other"


class TrainingLogger:
    """
    Custom callback that logs extended training metrics to JSON.

    Captures:
    - Gradient L2 norms, max abs, mean abs (grouped by backbone/neck/head)
    - Weight L2 norms per group
    - Per-class AP at each validation epoch
    - Wall-clock timestamps per epoch
    """

    def __init__(self, save_dir: str, config: dict = None):
        self.save_dir = save_dir
        self.metrics_dir = os.path.join(save_dir, "metrics")
        self.checkpoint_dir = os.path.join(save_dir, "weights", "epoch_checkpoints")
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.start_time = time.time()
        self.epoch_start_time = None

        self.history = {
            "experiment": config.get("experiment_name", "unknown") if config else "unknown",
            "model": config.get("model", {}).get("architecture", "unknown") if config else "unknown",
            "config": config,
            "total_epochs_run": 0,
            "early_stopped": False,
            "best_epoch": 0,
            "training_time_seconds": 0,
            "train_losses": [],
            "val_metrics": [],
            "per_class_ap": [],
            "learning_rates": [],
            "gradients": [],
            "weight_norms": [],
            "epoch_durations_seconds": [],
        }

        # Track epoch timing
        self.epoch_start = None

    def on_train_epoch_start(self, trainer):
        """Called at the start of each training epoch."""
        self.epoch_start = time.time()

    def on_train_epoch_end(self, trainer):
        """
        Called after training batches complete, before validation.
        Gradients are available here (after backward, before optimizer.zero_grad).
        """
        epoch = trainer.epoch

        # ── Training losses ──
        loss_items = {}
        if trainer.loss_items is not None:
            loss_names = trainer.loss_names if hasattr(trainer, 'loss_names') else ["box_loss", "cls_loss", "dfl_loss"]
            for name, val in zip(loss_names, trainer.loss_items):
                loss_items[name] = float(val)
        loss_items["epoch"] = epoch
        self.history["train_losses"].append(loss_items)

        # ── Learning rates ──
        lr_dict = {"epoch": epoch}
        if hasattr(trainer, 'lf') and hasattr(trainer, 'optimizer'):
            for i, pg in enumerate(trainer.optimizer.param_groups):
                lr_dict[f"pg{i}"] = float(pg.get("lr", 0))
        self.history["learning_rates"].append(lr_dict)

        # ── Gradient norms (grouped by backbone/neck/head) ──
        grad_stats = {"epoch": epoch}
        group_grads = defaultdict(list)

        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                group = _get_layer_group(name)
                group_grads[group].append(param.grad.detach().flatten())

        for group_name in ["backbone", "neck", "head"]:
            if group_grads[group_name]:
                all_grads = torch.cat(group_grads[group_name])
                grad_stats[group_name] = {
                    "l2_norm": float(torch.norm(all_grads, 2)),
                    "max_abs": float(torch.max(torch.abs(all_grads))),
                    "mean_abs": float(torch.mean(torch.abs(all_grads))),
                    "num_params": int(all_grads.numel()),
                }
            else:
                grad_stats[group_name] = {
                    "l2_norm": 0.0, "max_abs": 0.0, "mean_abs": 0.0, "num_params": 0
                }

        self.history["gradients"].append(grad_stats)

        # ── Weight norms ──
        weight_stats = {"epoch": epoch}
        group_weights = defaultdict(list)

        for name, param in trainer.model.named_parameters():
            group = _get_layer_group(name)
            group_weights[group].append(param.detach().flatten())

        for group_name in ["backbone", "neck", "head"]:
            if group_weights[group_name]:
                all_weights = torch.cat(group_weights[group_name])
                weight_stats[f"{group_name}_l2"] = float(torch.norm(all_weights, 2))
            else:
                weight_stats[f"{group_name}_l2"] = 0.0

        self.history["weight_norms"].append(weight_stats)

    def on_fit_epoch_end(self, trainer):
        """
        Called after training + validation for each epoch.
        Validation metrics are available here.
        """
        epoch = trainer.epoch

        # ── Epoch duration ──
        if self.epoch_start is not None:
            duration = time.time() - self.epoch_start
            self.history["epoch_durations_seconds"].append(duration)

        # ── Validation metrics ──
        val_dict = {"epoch": epoch}
        if hasattr(trainer, 'metrics') and trainer.metrics:
            metrics = trainer.metrics
            val_dict["precision"] = float(metrics.get("metrics/precision(B)", 0))
            val_dict["recall"] = float(metrics.get("metrics/recall(B)", 0))
            val_dict["mAP50"] = float(metrics.get("metrics/mAP50(B)", 0))
            val_dict["mAP50_95"] = float(metrics.get("metrics/mAP50-95(B)", 0))
        self.history["val_metrics"].append(val_dict)

        # ── Per-class AP ──
        per_class = {"epoch": epoch}
        if hasattr(trainer, 'validator') and trainer.validator is not None:
            validator = trainer.validator
            if hasattr(validator, 'metrics') and hasattr(validator.metrics, 'box'):
                box_metrics = validator.metrics.box
                names = trainer.model.names if hasattr(trainer.model, 'names') else {}

                if hasattr(box_metrics, 'maps') and hasattr(box_metrics, 'ap50'):
                    for i, cls_name in names.items():
                        try:
                            per_class[cls_name] = {
                                "ap50": float(box_metrics.ap50[i]) if i < len(box_metrics.ap50) else 0.0,
                                "ap50_95": float(box_metrics.maps[i]) if i < len(box_metrics.maps) else 0.0,
                            }
                        except (IndexError, AttributeError):
                            per_class[cls_name] = {"ap50": 0.0, "ap50_95": 0.0}

        self.history["per_class_ap"].append(per_class)

        # ── Track best epoch ──
        if trainer.best_fitness == trainer.fitness:
            self.history["best_epoch"] = epoch

        self.history["total_epochs_run"] = epoch + 1

        # ── Flush to disk (atomic write) ──
        self._save_to_disk()

    def on_model_save(self, trainer):
        """
        Called when best.pt or last.pt is saved.
        We save periodic checkpoints here.
        """
        epoch = trainer.epoch
        save_period = 10  # save every 10 epochs

        if (epoch + 1) % save_period == 0 and hasattr(trainer, 'last'):
            if trainer.last and os.path.exists(str(trainer.last)):
                dst = os.path.join(self.checkpoint_dir, f"epoch_{epoch + 1}.pt")
                shutil.copy2(str(trainer.last), dst)

    def on_train_end(self, trainer):
        """Called once after training completes."""
        self.history["training_time_seconds"] = time.time() - self.start_time
        self.history["early_stopped"] = trainer.epoch < (trainer.epochs - 1)

        self._save_to_disk()

        # Print summary
        print("\n" + "=" * 60)
        print("  TRAINING SUMMARY (Custom Logger)")
        print("=" * 60)
        print(f"  Total epochs:    {self.history['total_epochs_run']}")
        print(f"  Best epoch:      {self.history['best_epoch']}")
        print(f"  Early stopped:   {self.history['early_stopped']}")
        print(f"  Training time:   {self.history['training_time_seconds']:.0f}s "
              f"({self.history['training_time_seconds']/60:.1f} min)")

        # Best metrics
        if self.history["val_metrics"]:
            best_epoch = self.history["best_epoch"]
            for vm in self.history["val_metrics"]:
                if vm["epoch"] == best_epoch:
                    print(f"  Best mAP@50:     {vm.get('mAP50', 0):.4f}")
                    print(f"  Best mAP@50-95:  {vm.get('mAP50_95', 0):.4f}")
                    break

        print(f"\n  Logs saved to: {self.metrics_dir}/custom_metrics.json")
        print("=" * 60)

    def _save_to_disk(self):
        """Atomic write: write to .tmp, then rename."""
        path = os.path.join(self.metrics_dir, "custom_metrics.json")
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(self.history, f, indent=2, default=str)
            os.replace(tmp_path, path)
        except Exception as e:
            print(f"WARNING: Failed to save custom_metrics.json: {e}")

    def register(self, model):
        """Convenience method to register all callbacks at once."""
        model.add_callback("on_train_epoch_start", self.on_train_epoch_start)
        model.add_callback("on_train_epoch_end", self.on_train_epoch_end)
        model.add_callback("on_fit_epoch_end", self.on_fit_epoch_end)
        model.add_callback("on_model_save", self.on_model_save)
        model.add_callback("on_train_end", self.on_train_end)
