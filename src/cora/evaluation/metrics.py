"""
Evaluation metrics for CORA: BLEU-4, CIDEr, METEOR, ROUGE-L, SPICE.

Wraps pycocoevalcap for COCO-style captioning evaluation.
Supports CSV output and graceful SPICE fallback when Java is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Regex to strip emoji / surrogate-pair characters that crash Java SPICE parser.
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002702-\U000027B0"  # dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero-width joiner
    "\U000020E3"             # combining enclosing keycap
    "]+",
    flags=re.UNICODE,
)

try:
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    _HAS_COCO = True
except ImportError:
    _HAS_COCO = False


class CORAEvaluator:
    """Compute standard VLM captioning metrics (BLEU, CIDEr, METEOR, ROUGE-L, SPICE)."""

    METRIC_KEYS = ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE")

    def __init__(self) -> None:
        if not _HAS_COCO:
            logger.warning(
                "pycocoevalcap / pycocotools not installed. "
                "Install with: pip install pycocoevalcap pycocotools"
            )

    # ── Core evaluate ───────────────────────────────────────────────

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Compute metrics from parallel lists of prediction/reference strings.

        Returns dict with keys like 'bleu_4', 'cider', 'meteor', 'rouge_l', 'spice'.
        """
        if not _HAS_COCO:
            logger.error("Cannot compute metrics: pycocoevalcap not installed.")
            return {}

        if len(predictions) != len(references):
            raise ValueError(
                f"Predictions ({len(predictions)}) and references ({len(references)}) must have equal length."
            )

        # Sanitize: strip emoji / non-BMP chars that crash Java SPICE parser
        predictions = [_EMOJI_RE.sub("", p).strip() for p in predictions]
        references = [_EMOJI_RE.sub("", r).strip() for r in references]

        # Build COCO-format JSON structures
        pred_records = []
        ann_images = []
        ann_records = []

        for idx, (pred, ref) in enumerate(zip(predictions, references)):
            img_id = idx
            pred_records.append({"image_id": img_id, "caption": str(pred)})
            ann_images.append({"id": img_id})
            ann_records.append({"image_id": img_id, "id": idx, "caption": str(ref)})

        coco_ann = {
            "images": ann_images,
            "annotations": ann_records,
            "type": "captions",
            "info": {},
            "licenses": [],
        }

        with tempfile.TemporaryDirectory() as tmp:
            ann_path = os.path.join(tmp, "ann.json")
            res_path = os.path.join(tmp, "res.json")

            with open(ann_path, "w") as f:
                json.dump(coco_ann, f)
            with open(res_path, "w") as f:
                json.dump(pred_records, f)

            try:
                coco = COCO(ann_path)
                coco_res = coco.loadRes(res_path)
                evaluator = COCOEvalCap(coco, coco_res)
                evaluator.params["image_id"] = coco_res.getImgIds()
                evaluator.evaluate()
                raw = evaluator.eval
            except Exception as exc:
                logger.warning("Full COCO evaluation failed: %s", exc)
                logger.info("Retrying without SPICE (requires Java)...")
                try:
                    raw = self._evaluate_without_spice(ann_path, res_path)
                except Exception as exc2:
                    logger.error("Fallback evaluation also failed: %s", exc2)
                    return {}

        # Normalize keys
        result: Dict[str, float] = {}
        key_map = {
            "Bleu_1": "bleu_1",
            "Bleu_2": "bleu_2",
            "Bleu_3": "bleu_3",
            "Bleu_4": "bleu_4",
            "METEOR": "meteor",
            "ROUGE_L": "rouge_l",
            "CIDEr": "cider",
            "SPICE": "spice",
        }
        for coco_key, norm_key in key_map.items():
            val = raw.get(coco_key)
            if val is not None:
                result[norm_key] = float(val)
            else:
                # SPICE often fails without Java
                if norm_key == "spice":
                    logger.warning("SPICE metric unavailable (Java dependency). Skipping.")
                result[norm_key] = float("nan")

        return result

    # ── Fallback: evaluate without SPICE ─────────────────────────────

    @staticmethod
    def _evaluate_without_spice(ann_path: str, res_path: str) -> Dict[str, Any]:
        """Run individual COCO metrics skipping SPICE (which needs Java)."""
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        # NOTE: Meteor intentionally excluded — its Java subprocess deadlocks on large
        # datasets (5000+ samples) via stdin/stdout pipe overflow. CIDEr is our primary
        # metric; METEOR can be computed separately if needed.
        coco = COCO(ann_path)
        coco_res = coco.loadRes(res_path)
        img_ids = coco_res.getImgIds()

        # Build gts / res dicts in COCOEvalCap format
        gts: Dict[int, List[str]] = {}
        res: Dict[int, List[str]] = {}
        for img_id in img_ids:
            gts[img_id] = [ann["caption"] for ann in coco.imgToAnns[img_id]]
            res[img_id] = [ann["caption"] for ann in coco_res.imgToAnns[img_id]]

        scorers: list = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # Meteor() omitted — Java subprocess deadlocks on large datasets
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]


        raw: Dict[str, float] = {}
        for scorer, method in scorers:
            try:
                score, _ = scorer.compute_score(gts, res)
                if isinstance(method, list):
                    for sc, m in zip(score, method):
                        raw[m] = float(sc)
                else:
                    raw[method] = float(score)
            except Exception as exc:
                logger.warning("Scorer %s failed: %s", method, exc)

        logger.info("Computed %d metrics (without SPICE)", len(raw))
        return raw

    # ── Evaluate + CSV save ─────────────────────────────────────────

    def evaluate_and_save(
        self,
        predictions: List[str],
        references: List[str],
        csv_path: Union[str, Path],
        experiment_name: Optional[str] = None,
        precomputed: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Evaluate (or reuse *precomputed* metrics) and write summary CSV + JSON.

        Parameters
        ----------
        precomputed : dict, optional
            If provided, skip re-evaluation and save these metrics directly.
            This avoids a second SPICE invocation when metrics were already
            computed via :meth:`evaluate`.
        """
        import csv

        metrics = precomputed if precomputed is not None else self.evaluate(predictions, references)
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Summary CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["experiment"] + list(metrics.keys()) if experiment_name else list(metrics.keys())
            writer.writerow(header)
            row = [experiment_name] + [f"{v:.6f}" for v in metrics.values()] if experiment_name else [f"{v:.6f}" for v in metrics.values()]
            writer.writerow(row)

        # Also save as JSON for easier programmatic access
        json_path = csv_path.with_suffix(".json")
        with open(json_path, "w") as f:
            import json as _json
            _json.dump(metrics, f, indent=2)
        logger.info("Metrics saved to %s (+ %s)", csv_path, json_path)

        return metrics

    # ── Pretty print ────────────────────────────────────────────────

    @staticmethod
    def print_summary(metrics: Dict[str, float]) -> None:
        """Print metrics in a readable table."""
        print("\n" + "=" * 50)
        print("  Evaluation Results")
        print("=" * 50)
        for key, val in metrics.items():
            arrow = "\u2191"  # ↑
            print(f"  {key:<12s} ({arrow}): {val:>10.6f}")
        print("=" * 50 + "\n")

    # ── Simple accuracy ─────────────────────────────────────────────

    @staticmethod
    def simple_accuracy(predictions: List[str], references: List[str]) -> float:
        """Exact-match accuracy."""
        correct = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
        return correct / len(predictions) if predictions else 0.0
