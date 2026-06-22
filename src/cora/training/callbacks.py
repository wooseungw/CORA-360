"""PyTorch Lightning callbacks for CORA training.

Ported from legacy/CORA/cora/training/callbacks.py with enhancements:
  - Config YAML auto-copy alongside checkpoints
  - JSON metadata (epoch, global_step, val_loss, timestamp)
  - Periodic CUDA cache cleanup to prevent OOM from memory fragmentation
"""

from __future__ import annotations

import gc
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import lightning as pl

logger = logging.getLogger(__name__)


class MetadataCallback(pl.Callback):
    """Save training metadata alongside checkpoints.

    On every ``on_save_checkpoint`` event the callback:
      1. Writes / updates ``checkpoint_metadata.json`` in *ckpt_dir*.
      2. Copies the source YAML config once (idempotent).
    """

    def __init__(
        self,
        ckpt_dir: str,
        metadata: Dict[str, Any],
        config_path: Optional[str] = None,
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.metadata = metadata
        self.meta_path = self.ckpt_dir / "checkpoint_metadata.json"

        self.config_path = Path(config_path).resolve() if config_path else None
        self._config_copied = False

        if self.config_path:
            self.metadata.setdefault("config", {})
            self.metadata["config"]["source_path"] = str(self.config_path)

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_config_copy()

    # -- helpers --------------------------------------------------------------

    def _ensure_config_copy(self) -> None:
        if not self.config_path or self._config_copied:
            return
        try:
            if not self.config_path.exists():
                return
            target = self.ckpt_dir / "config.yaml"
            if not target.exists():
                shutil.copy2(self.config_path, target)
                logger.info("Copied config → %s", target)
            self._config_copied = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to copy config: %s", exc)

    # -- Lightning hooks ------------------------------------------------------

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        current = self.metadata.copy()
        current.update(
            {
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "val_loss": float(
                    trainer.callback_metrics.get("val_loss", 0.0)
                ),
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        try:
            with open(self.meta_path, "w") as fh:
                json.dump(current, fh, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save metadata JSON: %s", exc)

        self._ensure_config_copy()


class PreValidationCheckpoint(pl.Callback):
    """Save a checkpoint after training ends but BEFORE validation runs.

    Lightning's ``ModelCheckpoint`` saves only after validation (it needs
    ``val_loss`` for the filename).  If validation OOMs the checkpoint is
    lost and the entire stage must be re-trained.

    This callback saves a ``{stage}-pre_val.ckpt`` at ``on_train_epoch_end``
    so that a usable checkpoint always exists.  After validation completes
    successfully the ``ModelCheckpoint`` saves the canonical checkpoint with
    ``val_loss`` in the name, and ``CORATrainer._run_stage`` prefers that.
    """

    def __init__(self, stage_dir: str, stage: str) -> None:
        self.stage_dir = Path(stage_dir)
        self.stage = stage

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        save_path = self.stage_dir / f"{self.stage}-pre_val.ckpt"
        try:
            trainer.save_checkpoint(str(save_path))
            logger.info("Pre-validation checkpoint saved: %s", save_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save pre-validation checkpoint: %s", exc)


class CUDACacheCleanupCallback(pl.Callback):
    """Periodically clear CUDA cache and run GC to prevent memory fragmentation.

    Long-running training (7000+ steps) on bf16-mixed with LoRA can
    fragment CUDA memory.  This callback calls ``torch.cuda.empty_cache()``
    every *interval* training steps to release unused cached blocks.
    """

    def __init__(self, interval: int = 500) -> None:
        self.interval = interval

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.interval > 0 and (batch_idx + 1) % self.interval == 0:
            torch.cuda.empty_cache()
            gc.collect()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Periodically clear CUDA cache during validation to prevent OOM
        # from accumulated forward-pass memory fragmentation.
        if (batch_idx + 1) % 100 == 0:
            torch.cuda.empty_cache()

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        # Always clean before validation to free training-only buffers
        torch.cuda.empty_cache()
        gc.collect()
