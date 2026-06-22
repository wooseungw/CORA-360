"""Stage 1 — Vision-only SSL pretraining for panorama-aware features.

Trains a LoRA scoped to ``InternVLVisionModel`` (the ``vision_tower``) plus
a learnable ``PanoramaYawRoPE`` head, with the overlap VICReg-batchwise loss
as the sole objective. The language model is never invoked: forward runs
through ``InternVLModel.get_image_features`` (which internally does
``vision_tower → drop CLS → pixel_shuffle → multi_modal_projector``), then
``yaw_rope`` is applied per-sample on the resulting ``[B*V, 256, 1536]``
tensor before the loss.

Outputs per epoch:
    checkpoint-{epoch}/
        adapter_model.safetensors          (vision LoRA, via PEFT)
        adapter_config.json
        panoadapt_yaw_rope.pt              (learned RoPE frequencies)
    stage1_history.json                    (per-epoch inv/var/cov train+eval)

The contract with Stage 2: load the vision LoRA + yaw_rope, merge the vision
LoRA into base weights (or keep frozen as a second adapter), then attach a
new LoRA on the language_model only.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from PIL import Image, ImageFile
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers.trainer_utils import set_seed

from .config import BaselineConfig, PanoViewConfig
from .models import BaselineModelRegistry

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# InternVL model navigation
# ---------------------------------------------------------------------------


def _resolve_internvl_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the ``InternVLModel`` instance that owns ``get_image_features``.

    Works whether the top-level handle is the raw
    ``InternVLForConditionalGeneration`` or a PEFT-wrapped ``PeftModel``.
    """
    m = model
    while not hasattr(m, "get_image_features"):
        if hasattr(m, "base_model") and m.base_model is not m:
            m = m.base_model
            continue
        if hasattr(m, "model") and m.model is not m:
            m = m.model
            continue
        raise RuntimeError(
            "Stage1: could not locate InternVLModel.get_image_features on the "
            "given model. Stage 1 is currently wired only for InternVL3.x-hf."
        )
    return m


# ---------------------------------------------------------------------------
# Dataset — images + pano_meta only, no text/labels
# ---------------------------------------------------------------------------


class Stage1Dataset(torch.utils.data.Dataset):
    """CSV-backed dataset yielding per-sample ``pixel_values`` + ``pano_meta``.

    For each row:
        1. load the ERP panorama image
        2. (train mode) sample a (hfov, overlap) pair uniformly from
           ``hfov_deg_range``/``overlap_range``; eval uses static defaults
        3. build the multi-view pack via ``build_anyres_from_erp``
        4. push the resulting PIL views through the model's image_processor
        5. return ``{"pixel_values": [V, C, H, W], "pano_meta": dict}``

    No captions, no labels. Stage 1 needs only the images.
    """

    def __init__(
        self,
        csv_path: str,
        processor: Any,
        pano_view_config: PanoViewConfig,
        image_column: str = "url",
        max_samples: Optional[int] = None,
        augment: bool = False,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        if max_samples is not None and max_samples > 0:
            self.df = self.df.head(max_samples)
        self.processor = processor
        self.image_processor = processor.image_processor
        self.pano_view_config = pano_view_config
        self.image_column = image_column
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.df)

    def _build_views(self, erp_img: Image.Image) -> Tuple[List[Image.Image], Dict[str, Any]]:
        from cora.processors.anyres_e2p import build_anyres_from_erp

        cfg = self.pano_view_config
        hfov, overlap = cfg.hfov_deg, cfg.overlap
        if self.augment:
            if cfg.hfov_deg_range:
                lo, hi = float(cfg.hfov_deg_range[0]), float(cfg.hfov_deg_range[1])
                hfov = random.uniform(lo, hi)
            if cfg.overlap_range:
                lo, hi = float(cfg.overlap_range[0]), float(cfg.overlap_range[1])
                overlap = random.uniform(lo, hi)

        pack = build_anyres_from_erp(
            erp_img=erp_img,
            base_size=cfg.base_size,
            tile_render_size=cfg.tile_render_size,
            vit_size=cfg.vit_size,
            hfov_deg=hfov,
            overlap=overlap,
            closed_loop_yaw=cfg.closed_loop_yaw,
            pitch_min=cfg.pitch_min,
            pitch_max=cfg.pitch_max,
        )

        def _t2pil(t: torch.Tensor) -> Image.Image:
            arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return Image.fromarray(arr)

        views: List[Image.Image] = []
        if cfg.include_global:
            views.append(_t2pil(pack.global_image))
        for i in range(pack.tiles.size(0)):
            views.append(_t2pil(pack.tiles[i]))

        meta: Dict[str, Any] = dict(pack.yaw_geometry)
        meta["include_global"] = bool(cfg.include_global)
        meta["num_views"] = len(views)
        return views, meta

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        erp = Image.open(str(row[self.image_column])).convert("RGB")
        views, meta = self._build_views(erp)
        out = self.image_processor(views, return_tensors="pt")
        # InternVL image_processor stacks views along dim 0 → [V, C, H, W].
        return {"pixel_values": out["pixel_values"], "pano_meta": meta}


# ---------------------------------------------------------------------------
# Padded collate — variable per-sample V → pad to max(V) in batch
# ---------------------------------------------------------------------------


def stage1_collate_padded(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad pixel_values to ``max(V)`` per batch.

    Returns
    -------
    pixel_values : ``[B * max_V, C, H, W]`` flat for ``get_image_features``
    n_views_per_sample : ``[B]`` long  — number of REAL views per sample
    max_n_views : int                   padded view count per sample
    pano_meta_list : List[dict]         per-sample geometry
    """
    bs = len(features)
    max_v = max(f["pixel_values"].shape[0] for f in features)
    C, H, W = features[0]["pixel_values"].shape[1:]
    dtype = features[0]["pixel_values"].dtype

    padded = torch.zeros(bs, max_v, C, H, W, dtype=dtype)
    n_views = torch.zeros(bs, dtype=torch.long)
    metas: List[Dict[str, Any]] = []
    for i, f in enumerate(features):
        v = f["pixel_values"].shape[0]
        padded[i, :v] = f["pixel_values"]
        n_views[i] = v
        metas.append(f["pano_meta"])

    return {
        "pixel_values": padded.view(bs * max_v, C, H, W),
        "n_views_per_sample": n_views,
        "max_n_views": int(max_v),
        "pano_meta_list": metas,
    }


# ---------------------------------------------------------------------------
# Vision-only forward + yaw_rope chain
# ---------------------------------------------------------------------------


def stage1_forward(
    internvl_model: torch.nn.Module,
    pixel_values: torch.Tensor,
    n_views_per_sample: torch.Tensor,
    max_n_views: int,
    meta_list: List[Dict[str, Any]],
    yaw_rope: Optional[torch.nn.Module],
) -> List[torch.Tensor]:
    """``get_image_features`` + per-sample ``yaw_rope``.

    ``get_image_features`` returns ``[B*max_V, 256, 1536]`` for InternVL3-2B
    at 448 px (32×32 vit patches → 16×16 post pixel-shuffle, projected to
    LM hidden 1536). We split by sample, drop padded views, apply yaw_rope
    per real sample, and return the list of ``[V_i, 256, 1536]`` tensors.
    """
    feats = internvl_model.get_image_features(pixel_values=pixel_values)
    BV, T, D = feats.shape
    bs = n_views_per_sample.shape[0]
    if BV != bs * max_n_views:
        raise RuntimeError(
            f"Stage1: get_image_features returned BV={BV}, expected {bs}*{max_n_views}"
        )

    chunks: List[torch.Tensor] = []
    for i in range(bs):
        v = int(n_views_per_sample[i].item())
        start = i * max_n_views
        chunk = feats[start : start + v]  # [V_i, T, D]
        if yaw_rope is not None:
            yaw_rope.set_meta([meta_list[i]])
            chunk = yaw_rope(chunk, batch_size=1, num_views=v)
            yaw_rope.clear_meta()
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Overlap VICReg batchwise loss across all samples' real overlap pairs
# ---------------------------------------------------------------------------


@dataclass
class SSLComponents:
    invariance: float
    variance: float
    covariance: float
    total: float
    n_pairs: int


def compute_overlap_vicreg_batchwise(
    chunks: List[torch.Tensor],
    meta_list: List[Dict[str, Any]],
    include_global: bool,
    sim_w: float = 25.0,
    var_w: float = 25.0,
    cov_w: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, SSLComponents]:
    """Pool overlap tokens across all batch samples; compute batchwise VICReg.

    * invariance: per-pair MSE between adjacent tiles' overlap columns, then
      averaged over all pairs in the batch.
    * var/cov: computed on the GLOBAL pool of overlap tokens from EVERY pair
      in EVERY sample — that is what "batchwise" means under bs > 1.

    The global view (slot 0 when ``include_global``) is excluded from pair
    construction: it is the letterboxed full panorama, not a tile in the
    yaw cycle, so pairing it with view 1 would force an off-axis alignment.
    """
    all_curr: List[torch.Tensor] = []
    all_next: List[torch.Tensor] = []
    n_pairs_total = 0

    for chunk, meta in zip(chunks, meta_list):
        V, T, D = chunk.shape
        if V <= 1:
            continue
        side = int(math.isqrt(T))
        if side * side != T:
            raise RuntimeError(
                f"Stage1: non-square token grid T={T} (expected 16*16=256 for "
                f"InternVL3-2B@448). Got side={side}."
            )
        grid = chunk.view(V, side, side, D)

        if include_global:
            tiles = grid[1:]
        else:
            tiles = grid
        Vt = tiles.shape[0]
        if Vt <= 1:
            continue

        phys_overlap = float(meta["phys_overlap"])
        k = max(1, int(side * phys_overlap))

        curr_right = tiles[:, :, -k:, :]
        next_left = torch.roll(tiles, -1, dims=0)[:, :, :k, :]

        all_curr.append(curr_right.contiguous().view(Vt * side * k, D))
        all_next.append(next_left.contiguous().view(Vt * side * k, D))
        n_pairs_total += Vt

    if not all_curr:
        zero = torch.zeros((), device=chunks[0].device, requires_grad=True)
        return zero, SSLComponents(0.0, 0.0, 0.0, 0.0, 0)

    # Compute in float32 for numerical stability of var/cov under bf16 weights.
    curr_pool = torch.cat(all_curr, dim=0).float()
    next_pool = torch.cat(all_next, dim=0).float()

    inv = F.mse_loss(curr_pool, next_pool, reduction="mean")

    combined = torch.cat([curr_pool, next_pool], dim=0)  # [2N, D]
    std_all = torch.sqrt(combined.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(gamma - std_all).mean()

    cc = combined - combined.mean(dim=0, keepdim=True)
    cov = (cc.T @ cc) / max(combined.size(0) - 1, 1)
    cov_clone = cov.clone()
    cov_clone.diagonal().zero_()
    D = curr_pool.shape[-1]
    cov_loss = (cov_clone ** 2).sum() / D

    total = sim_w * inv + var_w * var_loss + cov_w * cov_loss
    comps = SSLComponents(
        invariance=float(inv.item()),
        variance=float(var_loss.item()),
        covariance=float(cov_loss.item()),
        total=float(total.item()),
        n_pairs=n_pairs_total,
    )
    return total.to(chunks[0].dtype), comps


# ---------------------------------------------------------------------------
# Vision LoRA wiring — scoped to vision_tower modules only
# ---------------------------------------------------------------------------


def _vision_lora_target_regex() -> str:
    """Single regex (re.fullmatch) selecting ``vision_tower`` Linear submodules.

    InternVL's vision encoder uses ``q_proj/k_proj/v_proj/projection_layer``
    in attention and ``fc1/fc2`` in MLP. The LM (Qwen2) reuses the leaf
    names ``q_proj/k_proj/v_proj/o_proj``, so plain suffix targets would
    bleed LoRA into the LM. PEFT applies ``re.fullmatch`` when
    ``target_modules`` is a STRING (not a list); a list falls back to suffix
    matching which can't disambiguate vision vs LM. So we pass a single
    fullmatch pattern anchored on the ``vision_tower`` path component.
    """
    leaf = r"(?:q_proj|k_proj|v_proj|projection_layer|fc1|fc2)"
    return rf".*vision_tower\..*\.{leaf}"


def attach_vision_lora(
    model: torch.nn.Module,
    r: int,
    alpha: int,
    dropout: float,
) -> torch.nn.Module:
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=_vision_lora_target_regex(),
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    return get_peft_model(model, cfg)


# ---------------------------------------------------------------------------
# Stage 1 Trainer (custom loop)
# ---------------------------------------------------------------------------


@dataclass
class Stage1Config:
    num_epochs: int = 30
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    seed: int = 42
    bf16: bool = True
    sim_w: float = 25.0
    var_w: float = 25.0
    cov_w: float = 1.0
    log_every_steps: int = 20
    save_every_epoch: bool = True
    dataloader_num_workers: int = 2
    gradient_checkpointing: bool = True


class Stage1Trainer:
    """Vision-only SSL pretraining loop."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Stage1Dataset,
        eval_dataset: Optional[Stage1Dataset],
        yaw_rope: Optional[torch.nn.Module],
        output_dir: Path,
        cfg: Stage1Config,
        include_global: bool,
    ) -> None:
        self.model = model
        self.internvl = _resolve_internvl_model(model)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.yaw_rope = yaw_rope
        self.output_dir = output_dir
        self.cfg = cfg
        self.include_global = include_global
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_dtype = torch.bfloat16 if cfg.bf16 else torch.float32

        self.model.to(self.device)
        if cfg.gradient_checkpointing:
            # Enable on the vision_tower itself — the top-level PEFT model's
            # gradient_checkpointing_enable hooks the LM input embeddings,
            # but the LM is not in our forward graph. For vision_tower's
            # checkpoint to actually recompute (saving memory), the first
            # checkpointed segment must receive a grad-requiring tensor; the
            # frozen patch_embeddings output is grad-free by default, so we
            # add a forward hook that flips requires_grad on its output.
            self.internvl.vision_tower.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

            def _force_emb_grad(_mod, _inp, out):
                if isinstance(out, torch.Tensor):
                    out.requires_grad_(True)
                return out

            self._gc_hook = self.internvl.vision_tower.embeddings.register_forward_hook(
                _force_emb_grad,
            )
        if self.yaw_rope is not None:
            ref = next(self.internvl.multi_modal_projector.parameters())
            self.yaw_rope.to(device=self.device, dtype=ref.dtype)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if self.yaw_rope is not None:
            trainable += [p for p in self.yaw_rope.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "Stage1: no trainable parameters. Vision LoRA not attached or "
                "yaw_rope missing — check config."
            )
        self.optimizer = torch.optim.AdamW(
            trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )

        steps_per_epoch = max(1, len(train_dataset) // cfg.batch_size)
        opt_steps = steps_per_epoch * cfg.num_epochs // max(1, cfg.gradient_accumulation_steps)
        warmup_steps = max(1, int(opt_steps * cfg.warmup_ratio))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda s: min(1.0, s / warmup_steps) * max(
                0.0, 1.0 - max(0, s - warmup_steps) / max(1, opt_steps - warmup_steps),
            ),
        )
        self.total_opt_steps = opt_steps

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.dataloader_num_workers,
            collate_fn=stage1_collate_padded,
            drop_last=True,
        )
        self.eval_loader = (
            torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.dataloader_num_workers,
                collate_fn=stage1_collate_padded,
                drop_last=False,
            )
            if eval_dataset is not None
            else None
        )

        self.history: List[Dict[str, Any]] = []
        self.global_step = 0

    # ----- forward + loss -----------------------------------------------------

    def _ssl_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, SSLComponents]:
        pixel_values = batch["pixel_values"].to(self.device, dtype=self.amp_dtype, non_blocking=True)
        n_views = batch["n_views_per_sample"].to(self.device)
        max_v = batch["max_n_views"]
        metas = batch["pano_meta_list"]

        chunks = stage1_forward(
            self.internvl, pixel_values, n_views, max_v, metas, self.yaw_rope,
        )
        loss, comps = compute_overlap_vicreg_batchwise(
            chunks,
            metas,
            include_global=self.include_global,
            sim_w=self.cfg.sim_w,
            var_w=self.cfg.var_w,
            cov_w=self.cfg.cov_w,
        )
        return loss, comps

    # ----- epoch loop ---------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        self.model.train()
        if self.yaw_rope is not None:
            self.yaw_rope.train()

        accum = self.cfg.gradient_accumulation_steps
        for epoch in range(self.cfg.num_epochs):
            t0 = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            agg = {"inv": 0.0, "var": 0.0, "cov": 0.0, "total": 0.0, "n": 0}

            pbar = tqdm(
                self.train_loader,
                desc=f"epoch {epoch+1}/{self.cfg.num_epochs}",
                dynamic_ncols=True,
            )
            for step, batch in enumerate(pbar):
                loss, comps = self._ssl_step(batch)
                (loss / accum).backward()
                agg["inv"] += comps.invariance
                agg["var"] += comps.variance
                agg["cov"] += comps.covariance
                agg["total"] += comps.total
                agg["n"] += 1

                if (step + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.optimizer.param_groups[0]["params"], self.cfg.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                if (step + 1) % self.cfg.log_every_steps == 0:
                    n = max(1, agg["n"])
                    pbar.set_postfix(
                        inv=f"{agg['inv']/n:.4f}",
                        var=f"{agg['var']/n:.4f}",
                        cov=f"{agg['cov']/n:.4f}",
                        lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                    )

            summary = {
                "epoch": epoch + 1,
                "global_step": self.global_step,
                "train_inv": agg["inv"] / max(1, agg["n"]),
                "train_var": agg["var"] / max(1, agg["n"]),
                "train_cov": agg["cov"] / max(1, agg["n"]),
                "train_total": agg["total"] / max(1, agg["n"]),
                "epoch_time_s": time.time() - t0,
            }
            if self.eval_loader is not None:
                summary.update(self._evaluate())

            self.history.append(summary)
            logger.info("epoch %d: %s", epoch + 1, summary)
            self._save_history()
            if self.cfg.save_every_epoch:
                self._save_checkpoint(epoch + 1)

        self._save_checkpoint("final")
        return {"history": self.history}

    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        self.model.eval()
        if self.yaw_rope is not None:
            self.yaw_rope.eval()
        agg = {"inv": 0.0, "var": 0.0, "cov": 0.0, "total": 0.0, "n": 0}
        for batch in self.eval_loader:
            _, comps = self._ssl_step(batch)
            agg["inv"] += comps.invariance
            agg["var"] += comps.variance
            agg["cov"] += comps.covariance
            agg["total"] += comps.total
            agg["n"] += 1
        self.model.train()
        if self.yaw_rope is not None:
            self.yaw_rope.train()
        n = max(1, agg["n"])
        return {
            "eval_inv": agg["inv"] / n,
            "eval_var": agg["var"] / n,
            "eval_cov": agg["cov"] / n,
            "eval_total": agg["total"] / n,
        }

    # ----- save ---------------------------------------------------------------

    def _save_checkpoint(self, tag: Any) -> None:
        d = self.output_dir / f"checkpoint-{tag}"
        d.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(d))
        if self.yaw_rope is not None:
            torch.save(self.yaw_rope.state_dict(), d / "panoadapt_yaw_rope.pt")
        logger.info("Stage1 checkpoint → %s", d)

    def _save_history(self) -> None:
        with open(self.output_dir / "stage1_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run_stage1(
    cfg: BaselineConfig,
    train_csv: str,
    eval_csv: Optional[str],
    output_dir: Path,
    stage1_cfg: Stage1Config,
) -> Path:
    set_seed(stage1_cfg.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.model.model_type.lower() not in {"internvl", "internvl_chat"}:
        raise NotImplementedError(
            f"Stage1 currently only supports InternVL3-hf; got "
            f"model_type={cfg.model.model_type}."
        )

    model, processor, _ = BaselineModelRegistry.load_model(cfg.model)

    # Freeze LM + projector. We train vision_tower (via LoRA) + yaw_rope only.
    internvl = _resolve_internvl_model(model)
    for p in internvl.language_model.parameters():
        p.requires_grad = False
    for p in internvl.multi_modal_projector.parameters():
        p.requires_grad = False

    model = attach_vision_lora(
        model,
        r=cfg.lora.r,
        alpha=cfg.lora.alpha,
        dropout=cfg.lora.dropout,
    )
    model.print_trainable_parameters()

    yaw_rope: Optional[torch.nn.Module] = None
    pa = cfg.panoadapt
    if pa is not None and pa.yaw_rope_enabled:
        from cora.model.positional import PanoramaYawRoPE

        internvl_after_peft = _resolve_internvl_model(model)
        embed_dim = internvl_after_peft.multi_modal_projector.linear_2.out_features
        include_global = (
            cfg.effective_pano_view.include_global if cfg.effective_pano_view else True
        )
        yaw_rope = PanoramaYawRoPE(
            embed_dim=embed_dim,
            rope_dim=pa.yaw_rope_dim,
            init_temperature=pa.yaw_rope_init_temperature,
            include_global=include_global,
        )

    pv = cfg.effective_pano_view
    train_ds = Stage1Dataset(
        csv_path=train_csv,
        processor=processor,
        pano_view_config=pv,
        image_column=cfg.data.image_column,
        max_samples=cfg.data.max_train_samples,
        augment=True,
    )
    eval_ds = (
        Stage1Dataset(
            csv_path=eval_csv,
            processor=processor,
            pano_view_config=pv,
            image_column=cfg.data.image_column,
            max_samples=cfg.data.max_eval_samples,
            augment=False,
        )
        if eval_csv
        else None
    )

    trainer = Stage1Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        yaw_rope=yaw_rope,
        output_dir=output_dir,
        cfg=stage1_cfg,
        include_global=(pv.include_global if pv else True),
    )
    trainer.train()
    return output_dir
