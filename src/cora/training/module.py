"""PyTorch Lightning module for CORA 3-stage training.

Stages
------
1. **vision**    – VICReg self-supervised overlap loss.
                   Trains Resampler + VICReg Projector (+ optional vision blocks).
2. **resampler** – **Joint VICReg + LM loss** (bridge stage).
                   VICReg regularises the Resampler to preserve spatial consistency
                   while LM loss teaches the PanoramaProjector to align vision
                   tokens with the frozen LLM.
                   Trains Resampler (low lr) + PanoramaProjector.
                   VICReg Projector is frozen (pass-through for gradient).
3. **finetune**  – LM loss only.
                   Trains PanoramaProjector + LLM LoRA.
                   Resampler is frozen (its role is complete).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import lightning as pl
from transformers import get_linear_schedule_with_warmup

from cora.config.schema import CORAConfig, StageConfig
from cora.training.losses import DenseCLLoss, GlobalLocalLoss, PanoContrastiveLoss, VICRegLoss

logger = logging.getLogger(__name__)


class PanoramaTrainingModule(pl.LightningModule):
    """Lightning module wrapping :class:`PanoramaVLM` for stage-aware training.

    Parameters
    ----------
    config : CORAConfig
        Full experiment configuration.
    stage : str
        One of ``"vision"``, ``"resampler"``, ``"finetune"``.
    vision_trainable_blocks : int
        Number of trailing vision-encoder blocks to unfreeze (0 = all frozen,
        -1 = all unfrozen).
    """

    def __init__(
        self,
        config: CORAConfig,
        stage: str = "finetune",
        vision_trainable_blocks: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.model_config = config.models
        self.stage = stage
        self.vision_trainable_blocks = vision_trainable_blocks

        # Resolve per-stage config (uses StageConfig.default_for fallback)
        self.stage_config: StageConfig = config.get_stage_config(stage)

        # --- 1. Build model (lazy import to avoid circular deps) ---
        self.model = self._build_model()
        self._setup_vision_lora()

        # --- 2. LoRA flag ---
        self.use_lora = (
            stage == "finetune"
            and config.lora is not None
            and config.lora.use_lora
        )

        # --- 3. Freeze / unfreeze ---
        self._setup_freezing()

        # --- 4. Loss ---
        # Vision loss is used in vision (primary) and resampler (regularisation).
        self.contrastive_loss: PanoContrastiveLoss | None = None
        self.densecl_loss: DenseCLLoss | None = None
        self.vicreg_loss: VICRegLoss | None = None
        self._vision_loss_type: str = "none"

        if stage in ("vision", "resampler"):
            self.vicreg_loss_weight = self.stage_config.vicreg_loss_weight
            if self.vicreg_loss_weight > 0:
                vision_loss_type = getattr(
                    self.stage_config, "vision_loss_type", "vicreg",
                )

                vicreg_or = (
                    self.stage_config.vicreg_overlap_ratio
                    if self.stage_config.vicreg_overlap_ratio is not None
                    else config.image_processing.overlap_ratio
                )

                if vision_loss_type == "contrastive":
                    self.contrastive_loss = PanoContrastiveLoss(
                        overlap_ratio=config.image_processing.overlap_ratio,
                        tau_overlap=self.stage_config.contrastive_tau_overlap,
                        tau_tile=self.stage_config.contrastive_tau_tile,
                        tile_loss_weight=self.stage_config.contrastive_tile_weight,
                    )
                    self._vision_loss_type = "contrastive"
                elif vision_loss_type == "densecl":
                    self.densecl_loss = DenseCLLoss(
                        overlap_ratio=vicreg_or,
                        temperature=self.stage_config.densecl_temperature,
                    )
                    self._vision_loss_type = "densecl"
                else:
                    self.vicreg_loss = VICRegLoss(
                        similarity_weight=self.stage_config.vicreg_similarity_weight,
                        variance_weight=self.stage_config.vicreg_variance_weight,
                        covariance_weight=self.stage_config.vicreg_covariance_weight,
                        overlap_ratio=vicreg_or,
                        vicreg_mode=self.stage_config.vicreg_mode,
                    )
                    self._vision_loss_type = "vicreg"

            self.gl_loss_weight = self.stage_config.global_local_loss_weight
            if self.gl_loss_weight > 0:
                self.global_local_loss = GlobalLocalLoss(
                    loss_type=self.stage_config.global_local_loss_type,
                )
            else:
                self.global_local_loss = None
        else:
            self.vicreg_loss_weight = 0.0
            self.global_local_loss = None
            self.gl_loss_weight = 0.0

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self) -> torch.nn.Module:
        """Build the PanoramaVLM; import deferred to avoid circular imports."""
        # Try the new cora.model path first, fall back to legacy path
        try:
            from cora.model.vlm import PanoramaVLM
        except ImportError:
            from cora.models.vlm import PanoramaVLM  # type: ignore[no-redef]
        return PanoramaVLM(self.config)

    def _setup_vision_lora(self) -> None:
        if self.config.lora is None:
            return

        enabled_any_stage = any(
            self.config.get_stage_config(stage_name).vision_lora_enabled
            for stage_name in self.config.training.stages
        )
        if not enabled_any_stage:
            return

        vision_encoder = getattr(self.model, "vision_encoder", None)
        if vision_encoder is None or not hasattr(vision_encoder, "setup_lora"):
            logger.warning("Vision encoder does not support LoRA setup; skipping.")
            return

        target_modules = self.stage_config.vision_lora_target_modules
        if target_modules is None:
            for stage_name in self.config.training.stages:
                stage_cfg = self.config.get_stage_config(stage_name)
                if stage_cfg.vision_lora_target_modules:
                    target_modules = stage_cfg.vision_lora_target_modules
                    break

        applied = vision_encoder.setup_lora(
            rank=self.config.lora.rank,
            alpha=self.config.lora.alpha,
            dropout=self.config.lora.dropout,
            target_modules=target_modules,
        )
        if applied:
            logger.info(
                "Vision LoRA attached for stage '%s' (trainable only when explicitly unfrozen).",
                self.stage,
            )

    # ------------------------------------------------------------------
    # Freezing logic
    # ------------------------------------------------------------------

    def _setup_freezing(self) -> None:
        """Freeze / unfreeze parameters according to the current stage.

        Stage 1 (vision):
            Resampler 🔥  |  VICReg Proj 🔥  |  PanoProj ❄️  |  LLM ❄️
        Stage 2 (resampler):
            Resampler 🔥 (low lr)  |  VICReg Proj ❄️ (gradient pass-through)
            PanoProj 🔥  |  LLM ❄️
        Stage 3 (finetune):
            Resampler ❄️  |  PanoProj 🔥  |  LLM LoRA 🔥
        """
        # 1. Freeze everything
        self.model.requires_grad_(False)

        # 2. Stage-specific unfreezing
        if self.stage == "vision":
            self._unfreeze_vision(self.vision_trainable_blocks)
            if self.stage_config.vision_lora_enabled:
                self._unfreeze_vision_lora()
            self._unfreeze_if_exists("resampler")
            self._unfreeze_vicreg_or_projector()

        elif self.stage == "resampler":
            self._unfreeze_vision(self.vision_trainable_blocks)
            self._unfreeze_if_exists("resampler")       # VICReg-regularised
            self._unfreeze_if_exists("projector")        # PanoramaProjector
            # vicreg_projector stays ❄️ — used only for loss pass-through

        elif self.stage == "finetune":
            self._unfreeze_vision(self.vision_trainable_blocks)
            # Resampler stays ❄️ — spatial + alignment learning complete
            self._unfreeze_if_exists("projector")
            if self.use_lora:
                self._unfreeze_lora()
            # Enable gradient checkpointing to reduce peak VRAM
            # (LoRA on 7 target modules retains many activations for backward)
            self._enable_gradient_checkpointing()

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        pct = trainable / total if total else 0.0
        logger.info(
            "Stage '%s': trainable %s / %s (%.1f%%)",
            self.stage, f"{trainable:,}", f"{total:,}", pct * 100,
        )

    def _unfreeze_vision(self, blocks: int) -> None:
        if blocks == 0:
            return
        ve = getattr(self.model, "vision_encoder", None)
        if ve is None:
            return
        if hasattr(ve, "unfreeze_last_n_blocks"):
            ve.unfreeze_last_n_blocks(blocks)
        elif blocks == -1:
            ve.requires_grad_(True)
        else:
            logger.warning(
                "Vision encoder has no unfreeze_last_n_layers; skipping block-wise unfreeze."
            )

    def _unfreeze_vision_lora(self) -> None:
        count = 0
        for name, param in self.model.named_parameters():
            if "vision_encoder" in name and "lora" in name.lower():
                param.requires_grad_(True)
                count += 1
        logger.info("Unfrozen %d vision LoRA parameters", count)

    def _unfreeze_if_exists(self, attr: str) -> None:
        sub = getattr(self.model, attr, None)
        if sub is not None:
            sub.requires_grad_(True)

    def _unfreeze_vicreg_or_projector(self) -> None:
        vp = getattr(self.model, "vicreg_projector", None)
        if vp is not None:
            vp.requires_grad_(True)
        else:
            self._unfreeze_if_exists("projector")

    def _unfreeze_lora(self) -> None:
        count = 0
        for name, p in self.model.named_parameters():
            if "lora" in name.lower():
                p.requires_grad_(True)
                count += 1
        logger.info("Unfrozen %d LoRA parameters", count)

    def _enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on the language model to trade compute for VRAM."""
        lm = getattr(self.model, "language_model", None)
        if lm is None:
            return
        # For PEFT-wrapped models, access the base model
        base = getattr(lm, "base_model", lm)
        base = getattr(base, "model", base)
        if hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            # Required for gradient checkpointing with LoRA
            if hasattr(base, "enable_input_require_grads"):
                base.enable_input_require_grads()
            logger.info("Enabled gradient checkpointing on language model")

    # ------------------------------------------------------------------
    # Forward / steps
    # ------------------------------------------------------------------

    def forward(self, **kwargs: Any) -> Dict[str, Any]:
        return self.model(stage=self.stage, **kwargs)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self(**batch)
        loss = outputs.get("loss", 0.0)

        if "vicreg_features" in outputs:
            vision_l = self._compute_vision_loss(outputs)
            if vision_l is not None:
                loss = loss + self.vicreg_loss_weight * vision_l
                self.log("train_vicreg", vision_l, sync_dist=True)

            if (
                self.global_local_loss is not None
                and outputs.get("global_features") is not None
            ):
                gl_l = self.global_local_loss(
                    outputs["global_features"],
                    outputs["vicreg_features"],
                    outputs["batch_size"],
                    outputs["num_views"],
                )
                loss = loss + self.gl_loss_weight * gl_l
                self.log("train_gl", gl_l, sync_dist=True)

        lm_loss = outputs.get("loss")
        if isinstance(lm_loss, torch.Tensor):
            self.log("train_lm", lm_loss, sync_dist=True)

        loss = self._ensure_grad(loss)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self(**batch)
        loss = outputs.get("loss", 0.0)

        if "vicreg_features" in outputs:
            vision_l = self._compute_vision_loss(outputs)
            if vision_l is not None:
                loss = loss + self.vicreg_loss_weight * vision_l
                self.log("val_vicreg", vision_l, sync_dist=True)

            if (
                self.global_local_loss is not None
                and outputs.get("global_features") is not None
            ):
                gl_l = self.global_local_loss(
                    outputs["global_features"],
                    outputs["vicreg_features"],
                    outputs["batch_size"],
                    outputs["num_views"],
                )
                loss = loss + self.gl_loss_weight * gl_l
                self.log("val_gl", gl_l, sync_dist=True)

        lm_loss = outputs.get("loss")
        if isinstance(lm_loss, torch.Tensor):
            self.log("val_lm", lm_loss, sync_dist=True)

        # Heavy view-consistency metrics only on first 50 batches to prevent OOM.
        # val_loss is still computed on ALL batches for accurate checkpoint selection.
        if batch_idx < 50:
            self._compute_view_consistency_metrics(outputs, prefix="val")

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    # Vision loss dispatch (VICReg or contrastive)
    # ------------------------------------------------------------------

    def _compute_vision_loss(
        self, outputs: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        """Compute vision self-supervised loss (VICReg, contrastive, or DenseCL).

        - **contrastive**: two dropout-augmented projections → symmetric InfoNCE
          (overlap + optional within-tile).
        - **densecl**: single projection → symmetric InfoNCE on overlap only.
        - **vicreg**: invariance + variance + covariance on overlap strips.
        """
        if self.contrastive_loss is not None:
            z1 = outputs["vicreg_features"]
            resamp_feats = outputs.get("resampler_features")
            if resamp_feats is None:
                return None

            z2 = self.model.vicreg_projector(resamp_feats)

            z1 = F.normalize(z1.float(), dim=-1)
            z2 = F.normalize(z2.float(), dim=-1)

            result = self.contrastive_loss(
                z1, z2,
                batch_size=outputs["batch_size"],
                num_views=outputs["num_views"],
            )
            pfx = "train" if self.training else "val"
            self.log(f"{pfx}_overlap_loss", result["overlap_loss"], sync_dist=True)
            self.log(f"{pfx}_tile_loss", result["tile_loss"], sync_dist=True)
            return result["loss"]

        if self.densecl_loss is not None:
            return self.densecl_loss(
                outputs["vicreg_features"],
                outputs["batch_size"],
                outputs["num_views"],
            )

        if self.vicreg_loss is not None:
            return self.vicreg_loss(
                outputs["vicreg_features"],
                outputs["batch_size"],
                outputs["num_views"],
            )

        return None

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> Any:
        lr = self.stage_config.lr

        # Differential learning-rate groups
        groups: Dict[str, List[torch.nn.Parameter]] = {
            "vision": [], "resampler": [], "projector": [],
            "llm": [], "other": [],
        }
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "vision_encoder" in name:
                groups["vision"].append(p)
            elif "resampler" in name:
                groups["resampler"].append(p)
            elif "projector" in name:
                groups["projector"].append(p)
            elif "language_model" in name:
                groups["llm"].append(p)
            else:
                groups["other"].append(p)

        param_groups: List[Dict[str, Any]] = []
        _add = param_groups.append
        if groups["vision"]:
            _add({"params": groups["vision"], "lr": lr * 0.1, "weight_decay": 0.01})
        if groups["resampler"]:
            # Stage 2: lower lr — VICReg regularises drift, low lr adds safety
            resampler_lr = lr * 0.1 if self.stage == "resampler" else lr
            _add({"params": groups["resampler"], "lr": resampler_lr, "weight_decay": 0.05})
        if groups["projector"]:
            _add({"params": groups["projector"], "lr": lr, "weight_decay": 0.05})
        if groups["llm"]:
            _add({"params": groups["llm"], "lr": lr, "weight_decay": 0.01})
        if groups["other"]:
            _add({"params": groups["other"], "lr": lr})

        if not param_groups:
            # Fallback: at least one group so the optimizer doesn't error
            param_groups = [{"params": list(self.model.parameters()), "lr": lr}]

        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-8)

        # Linear warmup + linear decay schedule
        total_steps = self._estimate_total_steps()
        if total_steps > 0:
            warmup = int(0.1 * total_steps)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup,
                num_training_steps=total_steps,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        return optimizer

    # ------------------------------------------------------------------
    # View consistency metrics (logged during validation)
    # ------------------------------------------------------------------

    @staticmethod
    def _linear_cka(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Linear CKA (Centered Kernel Alignment) between two feature matrices.

        Args:
            X: ``[N, D1]`` — first representation.
            Y: ``[N, D2]`` — second representation (same N samples).

        Returns:
            Scalar CKA ∈ [0, 1].  1 = identical representational geometry.
        """
        X = X.float() - X.float().mean(dim=0, keepdim=True)
        Y = Y.float() - Y.float().mean(dim=0, keepdim=True)
        hsic_xy = (X.T @ Y).norm() ** 2
        hsic_xx = (X.T @ X).norm()
        hsic_yy = (Y.T @ Y).norm()
        return hsic_xy / (hsic_xx * hsic_yy + eps)

    @staticmethod
    def _effective_rank(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Effective rank of a feature matrix via singular-value entropy.

        Args:
            X: ``[N, D]`` feature matrix.

        Returns:
            Scalar effective rank ∈ [1, min(N, D)].
            Collapse → rank ≈ 1.  Rich features → rank ≈ D.
        """
        s = torch.linalg.svdvals(X.float())
        p = s / (s.sum() + eps)
        p = p.clamp(min=eps)
        entropy = -(p * p.log()).sum()
        return entropy.exp()

    # ------------------------------------------------------------------
    # Hungarian matching accuracy
    # ------------------------------------------------------------------

    @staticmethod
    def _hungarian_accuracy(
        tile_feats: torch.Tensor, B: int, T: int,
    ) -> Optional[torch.Tensor]:
        """Leave-one-out centroid Hungarian matching accuracy.

        For each image *b*, compute per-position centroids from all
        OTHER images, then match image *b*'s tiles to those centroids
        via the Hungarian algorithm.  This avoids self-reference bias
        that inflates accuracy when features are collapsed.

        - Collapsed features → uniform LOO centroids → random → acc ≈ 1/T.
        - Discriminative features → distinct centroids → acc ≈ 1.0.
        """
        if B < 2:
            return None

        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            return None

        S, D = tile_feats.shape[1], tile_feats.shape[2]
        per_view = F.normalize(
            tile_feats.view(B, T, S, D).mean(dim=2), dim=-1,
        )  # [B, T, D]

        total_sum = per_view.sum(dim=0)  # [T, D]

        accs: list[float] = []
        for b in range(min(B, 8)):
            loo_centroid = F.normalize(
                (total_sum - per_view[b]) / (B - 1), dim=-1,
            )  # [T, D]
            cost = -torch.mm(per_view[b], loo_centroid.t())  # [T, T]
            row_ind, col_ind = linear_sum_assignment(cost.float().cpu().numpy())
            accs.append(float((row_ind == col_ind).mean()))

        if accs:
            return torch.tensor(sum(accs) / len(accs), device=tile_feats.device)
        return None

    # ------------------------------------------------------------------
    # View consistency metrics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_view_consistency_metrics(  # noqa: C901
        self,
        outputs: Dict[str, Any],
        prefix: str = "val",
    ) -> None:
        """Compute and log spatial-consistency diagnostics for panoramic views.

        Logged metrics (all non-loss; eval-time only)
        -----------------------------------------------
        Post-projector (vicreg_features)
            ``{prefix}_adj_cos``          – cosine similarity in overlap zones  (↑)
            ``{prefix}_adj_mse``          – MSE in overlap zones               (↓)
            ``{prefix}_adj_cka``          – Linear CKA in overlap zones        (↑)
            ``{prefix}_feat_std``         – mean per-dim std (collapse ≈ 0)
            ``{prefix}_eff_rank``         – singular-value entropy rank         (↑)
            ``{prefix}_inter_view_cka``   – CKA between non-adjacent tiles     (moderate)

        Pre-projector (resampler_features) — same but with ``_r`` suffix
            ``{prefix}_r_eff_rank``       – resampler effective rank            (↑)
            ``{prefix}_r_inter_view_cka`` – resampler inter-view CKA           (moderate)

        Spatial discriminability
            ``{prefix}_hungarian_acc``    – Hungarian matching accuracy         (↑, random=1/T)

        Global–local alignment
            ``{prefix}_gl_cos``           – pooled global ↔ tiles cosine       (↑)
            ``{prefix}_gl_cka``           – Linear CKA  global ↔ tiles         (↑)
        """
        tile_feats = outputs.get("vicreg_features")
        if tile_feats is None:
            return  # nothing to measure (stage 3)

        B = outputs["batch_size"]
        T = outputs["num_views"]  # number of tiles (excluding global)

        if T <= 1:
            return

        # tile_feats: [B*T, S, D]
        S, D = tile_feats.shape[1], tile_feats.shape[2]
        H = W = int(math.isqrt(S))

        # Reshape to spatial grid [B, T, H, W, D]
        grid = tile_feats.view(B, T, H, W, D)

        # =============================================================
        # 1. Adjacent overlap: cosine sim, MSE, CKA
        #    Uses the PHYSICAL overlap (image_processing.overlap_ratio)
        #    to measure true spatial consistency.
        # =============================================================
        overlap_ratio = self.config.image_processing.overlap_ratio
        k = max(1, int(W * overlap_ratio))

        curr_right = grid[:, :-1, :, -k:, :]   # [B, T-1, H, k, D]
        next_left = grid[:, 1:, :, :k, :]       # [B, T-1, H, k, D]

        curr_flat = curr_right.reshape(-1, D)    # [B*(T-1)*H*k, D]
        next_flat = next_left.reshape(-1, D)

        cosine_sim = F.cosine_similarity(curr_flat, next_flat, dim=-1).mean()
        overlap_mse = F.mse_loss(curr_flat, next_flat)
        adj_cka = self._linear_cka(curr_flat, next_flat)

        self.log(f"{prefix}_adj_cos", cosine_sim, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}_adj_mse", overlap_mse, sync_dist=True)
        self.log(f"{prefix}_adj_cka", adj_cka, prog_bar=True, sync_dist=True)

        # =============================================================
        # 2. Representation health: std + effective rank (post-projector)
        # =============================================================
        feat_std = tile_feats.std(dim=-1).mean()
        self.log(f"{prefix}_feat_std", feat_std, sync_dist=True)

        pooled = tile_feats.mean(dim=1)  # [B*T, D]
        eff_rank = self._effective_rank(pooled)
        self.log(f"{prefix}_eff_rank", eff_rank, sync_dist=True)

        # =============================================================
        # 3. Global–local alignment: cosine + CKA
        # =============================================================
        global_feats = outputs.get("global_features")
        if global_feats is not None:
            g_pooled = global_feats.mean(dim=1)                        # [B, D]
            t_per_img = tile_feats.view(B, T, S, D).mean(dim=(1, 2))  # [B, D]

            g_norm = F.normalize(g_pooled, dim=-1)
            t_norm = F.normalize(t_per_img, dim=-1)
            gl_cosine = (g_norm * t_norm).sum(dim=-1).mean()
            gl_cka = self._linear_cka(g_pooled, t_per_img)

            self.log(f"{prefix}_gl_cos", gl_cosine, sync_dist=True)
            self.log(f"{prefix}_gl_cka", gl_cka, sync_dist=True)

        # =============================================================
        # 4. Inter-view diversity: CKA between non-adjacent tile pairs
        # =============================================================
        if T >= 3:
            per_view = tile_feats.view(B, T, S, D)
            cka_vals: list[torch.Tensor] = []
            step = max(2, T // 4)
            for i in range(0, T, step):
                j = (i + T // 2) % T
                if i == j:
                    continue
                vi = per_view[:, i].reshape(-1, D)  # [B*S, D]
                vj = per_view[:, j].reshape(-1, D)
                cka_vals.append(self._linear_cka(vi, vj))
            if cka_vals:
                inter_cka = torch.stack(cka_vals).mean()
                self.log(f"{prefix}_inter_view_cka", inter_cka, sync_dist=True)

        # =============================================================
        # 5. Hungarian matching accuracy (spatial discriminability)
        # =============================================================
        hung_acc = self._hungarian_accuracy(tile_feats, B, T)
        if hung_acc is not None:
            self.log(f"{prefix}_hungarian_acc", hung_acc, prog_bar=True, sync_dist=True)

        # =============================================================
        # 6. Overlap retrieval accuracy (contrastive quality metric)
        # =============================================================
        if T >= 2:
            per_view_size = H * k
            num_pairs = B * (T - 1)
            curr_norm = F.normalize(
                curr_flat.view(num_pairs, per_view_size, D).float(), dim=-1,
            )
            next_norm = F.normalize(
                next_flat.view(num_pairs, per_view_size, D).float(), dim=-1,
            )
            sims = torch.bmm(curr_norm, next_norm.transpose(1, 2))
            preds = sims.argmax(dim=-1)
            targets = torch.arange(
                per_view_size, device=tile_feats.device,
            ).unsqueeze(0).expand(num_pairs, -1)
            retrieval_acc = (preds == targets).float().mean()
            self.log(f"{prefix}_overlap_ret_acc", retrieval_acc, prog_bar=True, sync_dist=True)

        # =============================================================
        # 7. Pre-projector (resampler output) health — key collapse check
        #    The VICReg projector can "absorb" invariance while the
        #    backbone collapses; these metrics reveal whether the
        #    resampler output retains diversity.
        # =============================================================
        resamp_feats = outputs.get("resampler_features")
        if resamp_feats is not None and resamp_feats.shape[0] == B * T:
            S_r, D_r = resamp_feats.shape[1], resamp_feats.shape[2]
            r_pooled = resamp_feats.mean(dim=1)  # [B*T, D_r]
            self.log(f"{prefix}_r_eff_rank", self._effective_rank(r_pooled), sync_dist=True)

            # Inter-view CKA on resampler output
            if T >= 3:
                r_per_view = resamp_feats.view(B, T, S_r, D_r)
                r_cka_vals: list[torch.Tensor] = []
                r_step = max(2, T // 4)
                for i in range(0, T, r_step):
                    j = (i + T // 2) % T
                    if i == j:
                        continue
                    ri = r_per_view[:, i].reshape(-1, D_r)
                    rj = r_per_view[:, j].reshape(-1, D_r)
                    r_cka_vals.append(self._linear_cka(ri, rj))
                if r_cka_vals:
                    r_inter_cka = torch.stack(r_cka_vals).mean()
                    self.log(f"{prefix}_r_inter_cka", r_inter_cka, sync_dist=True)

            # Hungarian on resampler features
            r_hung = self._hungarian_accuracy(resamp_feats, B, T)
            if r_hung is not None:
                self.log(f"{prefix}_r_hungarian_acc", r_hung, sync_dist=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_grad(self, loss: Any) -> torch.Tensor:
        """Make sure ``loss`` is a differentiable tensor on the correct device."""
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(float(loss), device=self.device)
        # Attach to graph if detached (e.g. scalar 0.0 from model)
        anchor = next((p for p in self.model.parameters() if p.requires_grad), None)
        if anchor is not None and (not loss.requires_grad or loss.grad_fn is None):
            loss = loss + anchor.reshape(-1)[0] * 0.0
        return loss

    def _estimate_total_steps(self) -> int:
        """Best-effort estimate of total optimiser steps for the scheduler."""
        try:
            dm = self.trainer.datamodule
            if dm is not None:
                steps_per_epoch = len(dm.train_dataloader())
            else:
                steps_per_epoch = 100
        except Exception:
            steps_per_epoch = 100
        epochs = self.trainer.max_epochs if self.trainer.max_epochs else 1
        accum = self.stage_config.accumulate_grad_batches or 1
        return (steps_per_epoch * epochs) // accum
