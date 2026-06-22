"""CORA 3-stage training orchestrator.

Ported from legacy/CORA/cora/training/trainer.py with enhancements:
  - ``runs/{experiment}/{YYYYMMDD_NNN}/`` output layout via OutputConfig
  - ``stage_state.json`` for resuming multi-stage pipelines
  - Weight-only checkpoint loading between stages (reset optimiser)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import lightning as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

from cora.config.schema import CORAConfig, StageConfig
from cora.config.manager import ConfigManager
from cora.training.callbacks import PreValidationCheckpoint
from cora.training.module import PanoramaTrainingModule
from cora.training.callbacks import CUDACacheCleanupCallback, MetadataCallback

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage-state persistence helpers
# ---------------------------------------------------------------------------

_STAGE_STATE_FILE = "stage_state.json"


def _load_stage_state(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / _STAGE_STATE_FILE
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return {"completed_stages": [], "current_stage": None}


def _save_stage_state(run_dir: Path, state: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / _STAGE_STATE_FILE, "w") as fh:
        json.dump(state, fh, indent=2)


# ---------------------------------------------------------------------------
# CORATrainer
# ---------------------------------------------------------------------------


class CORATrainer:
    """Orchestrate the 3-stage CORA training pipeline.

    Stages are executed sequentially (``vision → resampler → finetune``).
    Between stages the *model weights* are carried forward but the optimiser
    state is reset.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    resume : str | None
        ``"auto"`` – resume from the last stage_state checkpoint.
        ``"path/to/ckpt"`` – explicit checkpoint path.
        ``None`` – start fresh.
    """

    def __init__(self, config_path: str, resume: Optional[str] = None) -> None:
        self.config: CORAConfig = ConfigManager.load(config_path)
        self.config_path = config_path

        exp_name = self.config.experiment.get("name", "unnamed")

        # Resume logic — when resuming, reuse the latest existing run
        # directory instead of creating a new one.
        self._resume_ckpt: Optional[str] = None
        if resume == "auto":
            existing = self._find_latest_run_dir(exp_name)
            if existing is not None:
                self.run_dir = existing
                logger.info("Resuming run directory: %s", self.run_dir)
            else:
                self.run_dir = self.config.output.resolve_experiment_dir(exp_name)
                logger.info("No existing run found — new directory: %s", self.run_dir)
        else:
            self.run_dir = self.config.output.resolve_experiment_dir(exp_name)
            logger.info("Run directory: %s", self.run_dir)

        # Stage state
        self._stage_state = _load_stage_state(self.run_dir)

        if resume == "auto":
            self._resume_ckpt = self._resolve_auto_resume()
        elif resume:
            self._resume_ckpt = resume

        # Processor (lazy — built on first use)
        self._processor: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, stage: Optional[str] = None) -> Optional[str]:
        """Run all configured stages and return the final best-checkpoint path.

        Parameters
        ----------
        stage : str | None
            If given, run only this single stage instead of the full pipeline.
        """
        stages: List[str] = [stage] if stage else self.config.training.stages
        logger.info("Training pipeline stages: %s", stages)

        ckpt = self._resume_ckpt
        for stage in stages:
            if stage in self._stage_state.get("completed_stages", []):
                prev = self._stage_state.get(stage, {}).get("checkpoint")
                if prev and os.path.exists(prev):
                    ckpt = prev
                logger.info("Stage '%s' already completed — skipping.", stage)
                continue

            logger.info("═══ Starting stage: %s ═══", stage)
            self._stage_state["current_stage"] = stage
            _save_stage_state(self.run_dir, self._stage_state)

            ckpt = self._run_stage(stage, resume_checkpoint=ckpt)

            self._stage_state.setdefault("completed_stages", []).append(stage)
            self._stage_state[stage] = {"checkpoint": ckpt}
            self._stage_state["current_stage"] = None
            _save_stage_state(self.run_dir, self._stage_state)

            logger.info("═══ Finished stage: %s ═══\n", stage)

        return ckpt

    # ------------------------------------------------------------------
    # Single-stage runner
    # ------------------------------------------------------------------

    def _run_stage(
        self,
        stage: str,
        resume_checkpoint: Optional[str] = None,
    ) -> Optional[str]:
        """Execute one training stage and return best checkpoint path."""
        stage_cfg: StageConfig = self.config.get_stage_config(stage)

        # --- 1. DataModule ---
        batch_size = stage_cfg.batch_size
        datamodule = self._build_datamodule(stage_cfg)

        # --- 2. Lightning module ---
        module = PanoramaTrainingModule(
            config=self.config,
            stage=stage,
            vision_trainable_blocks=stage_cfg.vision_trainable_blocks,
        )

        # Load weights from previous stage (without optimizer state)
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            logger.info("Loading weights from: %s", resume_checkpoint)
            ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            module.load_state_dict(state_dict, strict=False)

        # --- AutoBatch ---
        if batch_size == -1:
            batch_size = self._autobatch(module, datamodule, stage)
            datamodule = self._build_datamodule(stage_cfg, batch_size_override=batch_size)

        # --- 3. Callbacks ---
        stage_dir = self.run_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        ckpt_callback = ModelCheckpoint(
            dirpath=str(stage_dir),
            filename=f"{stage}-{{epoch:02d}}-{{val_loss:.4f}}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,
        )
        meta_callback = MetadataCallback(
            ckpt_dir=str(stage_dir),
            metadata={
                "stage": stage,
                "experiment": self.config.experiment.get("name", ""),
            },
            config_path=self.config_path,
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        cache_cleanup = CUDACacheCleanupCallback(interval=500)
        pre_val_ckpt = PreValidationCheckpoint(stage_dir=str(stage_dir), stage=stage)
        callbacks: List[pl.Callback] = [ckpt_callback, meta_callback, lr_monitor, cache_cleanup, pre_val_ckpt]

        # --- 4. Logger ---
        pl_logger = self._build_logger(stage)

        # --- 5. Trainer ---
        trainer = pl.Trainer(
            max_epochs=stage_cfg.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=self.config.training.devices,
            precision=self.config.training.precision,
            callbacks=callbacks,
            logger=pl_logger,
            gradient_clip_val=self.config.training.gradient_clip_val,
            accumulate_grad_batches=stage_cfg.accumulate_grad_batches,
            strategy=self.config.training.strategy,
            default_root_dir=str(self.run_dir),
        )

        # --- 6. Fit ---
        trainer.fit(module, datamodule=datamodule)

        best = ckpt_callback.best_model_path

        # Fallback: if ModelCheckpoint didn't save (e.g. validation OOM),
        # use the pre-validation checkpoint saved at end of training.
        if not best or not os.path.exists(best):
            pre_val_path = stage_dir / f"{stage}-pre_val.ckpt"
            if pre_val_path.exists():
                logger.warning(
                    "No validated checkpoint found for stage '%s'. "
                    "Using pre-validation checkpoint: %s",
                    stage, pre_val_path,
                )
                best = str(pre_val_path)

        logger.info("Best checkpoint for stage '%s': %s", stage, best)
        return best or None

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def _build_datamodule(
        self,
        stage_cfg: StageConfig,
        batch_size_override: Optional[int] = None,
    ) -> Any:
        """Build or return the DataModule for the current stage.

        Uses stage-specific CSV paths if provided, else falls back to
        top-level ``config.data``.
        """
        # Deferred import — data module may not exist yet during early porting
        try:
            from cora.data.datamodule import PanoramaDataModule
        except ImportError:
            logger.warning(
                "cora.data.dataset.PanoramaDataModule not found; "
                "returning None (manual datamodule required)."
            )
            return None

        # Resolve CSVs: stage-level → top-level
        stage_data = stage_cfg.data
        if stage_data and stage_data.csv_train:
            train_csv = stage_data.csv_train
            val_csv = stage_data.csv_val
        else:
            train_csv = self.config.data.get("train", self.config.data.get("train_csv", ""))
            val_csv = self.config.data.get("val", self.config.data.get("val_csv", ""))

        # Normalise to str (PanoramaDataModule/Dataset expects a single path)
        if isinstance(train_csv, list):
            train_csv = train_csv[0] if train_csv else ""
        if isinstance(val_csv, list):
            val_csv = val_csv[0] if val_csv else ""

        processor = self._get_processor()
        max_text_length = getattr(stage_cfg, "max_text_length", 4096)
        if isinstance(max_text_length, str):
            max_text_length = 4096

        bs = batch_size_override if batch_size_override is not None else stage_cfg.batch_size
        bs = max(bs, 1)

        return PanoramaDataModule(
            train_csv=train_csv,
            val_csv=val_csv,
            image_root=self.config.data.get("image_root"),
            processor=processor,
            batch_size=bs,
            num_workers=self.config.training.num_workers,
            max_text_length=max_text_length,
        )

    def _get_processor(self) -> Any:
        """Lazily build the PanoramaProcessor."""
        if self._processor is not None:
            return self._processor

        try:
            from cora.processors.images import PanoramaImageProcessor
            from cora.processors.processor import PanoramaProcessor
        except ImportError:
            logger.warning("Processors not available; returning None.")
            return None

        from transformers import AutoTokenizer

        img_cfg = self.config.image_processing
        image_size = img_cfg.image_size
        if image_size is None:
            image_size = self._infer_image_size()

        img_proc = PanoramaImageProcessor(
            image_size=image_size,
            crop_strategy=img_cfg.crop_strategy,
            fov_deg=img_cfg.fov_deg,
            overlap_ratio=img_cfg.overlap_ratio,
            normalize=img_cfg.normalize,
            use_vision_processor=img_cfg.use_vision_processor,
            vision_model_name=self.config.models.vision_name,
            anyres_max_patches=img_cfg.anyres_max_patches,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.models.language_model_name, trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Register <|vision|> special token so the data pipeline produces the
        # same token ID that LanguageFusion.fuse() expects from the model's
        # tokenizer.  Without this, <|vision|> is split into sub-tokens and
        # vision-embedding insertion falls back to a naive prefix mode.
        vision_token = "<|vision|>"
        existing = [str(t) for t in tokenizer.additional_special_tokens]
        if vision_token not in existing:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": existing + [vision_token]}
            )

        system_prompt = getattr(self.config.training, "system_msg", None) or None
        self._processor = PanoramaProcessor(img_proc, tokenizer, system_prompt=system_prompt)
        return self._processor

    def _infer_image_size(self) -> List[int]:
        """Heuristic: infer image size from vision model name."""
        vname = self.config.models.vision_name.lower()
        for tag, size in [
            ("-256", [256, 256]),
            ("-560", [560, 560]),
            ("-384", [384, 384]),
            ("-224", [224, 224]),
        ]:
            if tag in vname:
                logger.info("Auto-inferred image size %s from '%s'", size, vname)
                return size
        logger.warning("Could not infer image size from '%s'; defaulting to [224,224]", vname)
        return [224, 224]

    def _build_logger(self, stage: str) -> Optional[Any]:
        """Build W&B logger if configured."""
        project = self.config.training.wandb_project
        if not project:
            return None
        try:
            return WandbLogger(
                project=project,
                name=f"{self.config.experiment.get('name', 'cora')}-{stage}",
                config=self.config.model_dump(),
                save_dir=str(self.run_dir),
            )
        except Exception:
            logger.warning("W&B logger init failed; training without remote logging.")
            return None

    def _autobatch(
        self,
        module: PanoramaTrainingModule,
        datamodule: Any,
        stage: str,
    ) -> int:
        """Run GPU memory profiling to find optimal batch size for a stage."""
        from cora.training.autobatch import autobatch
        from cora.data.dataset import custom_collate_fn

        datamodule.setup("fit")
        dataset = datamodule.train_ds

        def _training_step(probe: torch.nn.Module, batch: dict) -> torch.Tensor:
            outputs = probe(**batch)
            loss = outputs.get("loss", 0.0)
            if probe.vicreg_loss is not None and "vicreg_features" in outputs:
                vicreg_l = probe.vicreg_loss(
                    outputs["vicreg_features"],
                    outputs["batch_size"],
                    outputs["num_views"],
                )
                loss = loss + probe.vicreg_loss_weight * vicreg_l
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(float(loss), device=next(probe.parameters()).device, requires_grad=True)
            anchor = next((p for p in probe.parameters() if p.requires_grad), None)
            if anchor is not None and not loss.requires_grad:
                loss = loss + anchor.reshape(-1)[0] * 0.0
            return loss

        optimal = autobatch(
            module=module,
            dataset=dataset,
            collate_fn=custom_collate_fn,
            fraction=0.85,
            default_batch_size=1,
            training_step_fn=_training_step,
        )
        logger.info("AutoBatch [%s]: batch_size = %d", stage, optimal)
        return optimal

    def _find_latest_run_dir(self, experiment_name: str) -> Optional[Path]:
        """Find the most recent existing run directory that has a stage_state.

        Scans ``runs/{experiment_name}/`` for directories with a valid
        ``stage_state.json`` containing at least one completed stage, and
        returns the latest one (by directory name sort order).  Returns
        ``None`` if no such directory exists.
        """
        base = Path(self.config.output.runs_dir) / experiment_name
        if not base.exists():
            return None
        candidates: list[Path] = []
        for d in sorted(base.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            state_file = d / _STAGE_STATE_FILE
            if state_file.exists():
                state = _load_stage_state(d)
                if state.get("completed_stages"):
                    candidates.append(d)
                    break  # sorted reverse — first match is latest
        return candidates[0] if candidates else None

    def _resolve_auto_resume(self) -> Optional[str]:
        """Find the latest checkpoint from stage_state for auto-resume."""
        completed = self._stage_state.get("completed_stages", [])
        if not completed:
            return None
        last = completed[-1]
        ckpt = self._stage_state.get(last, {}).get("checkpoint")
        if ckpt and os.path.exists(ckpt):
            logger.info("Auto-resume from stage '%s': %s", last, ckpt)
            return ckpt
        return None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CORA 3-stage trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="'auto' or explicit checkpoint path",
    )
    args = parser.parse_args()

    trainer = CORATrainer(args.config, resume=args.resume)
    trainer.train()
