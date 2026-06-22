#!/usr/bin/env python3
"""LoRA adapter merging via PEFT (TIES / DARE-TIES / linear).

Merges two independently-trained LoRA adapters and evaluates the result.

Usage examples:
    # TIES merge of SFT-only + DenseCL adapters (InternVL3.5-2B)
    python scripts/merge_lora.py \
        --config configs/baseline/panoadapt_internvl35_2b.yaml \
        --adapter-a runs/baseline/ablation_internvl35-2b_anyrese2p_noloss/lora_adapter \
        --adapter-b runs/baseline/panoadapt_internvl35-2b/lora_adapter \
        --method ties --density 0.5 \
        --output-dir runs/baseline/merged_ties_internvl35-2b

    # Sweep density values
    python scripts/merge_lora.py \
        --config configs/baseline/panoadapt_internvl35_2b.yaml \
        --adapter-a ... --adapter-b ... \
        --method ties --density 0.3 0.5 0.7 \
        --output-dir runs/baseline/merged_sweep_internvl35-2b
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def merge_adapters(
    config_path: str,
    adapter_a_path: str,
    adapter_b_path: str,
    output_path: str,
    method: str = "ties",
    density: float = 0.5,
    weight_a: float = 1.0,
    weight_b: float = 1.0,
) -> str:
    """Load base model via BaselineModelRegistry, attach two LoRA adapters, merge via PEFT, and save.

    Args:
        config_path: Path to baseline YAML config.
        adapter_a_path: Path to first LoRA adapter directory.
        adapter_b_path: Path to second LoRA adapter directory.
        output_path: Directory to save merged adapter.
        method: Combination type.
        density: Sparsity parameter for TIES/DARE.
        weight_a: Weight for adapter A in the merge.
        weight_b: Weight for adapter B in the merge.

    Returns:
        Path to saved merged adapter directory.
    """
    import yaml

    import torch
    from peft import PeftModel

    from cora.baseline.config import BaselineConfig
    from cora.baseline.models import BaselineModelRegistry

    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    config = BaselineConfig(**cfg_dict)

    logger.info("Loading base model: %s", config.model.hf_model_id)
    model, processor, tokenizer = BaselineModelRegistry.load_model(config.model)

    # Load first adapter
    logger.info("Loading adapter A: %s", adapter_a_path)
    model = PeftModel.from_pretrained(model, adapter_a_path, adapter_name="sft")

    # Load second adapter
    logger.info("Loading adapter B: %s", adapter_b_path)
    model.load_adapter(adapter_b_path, adapter_name="ssl")

    # Merge
    merge_name = f"merged_{method}_d{str(density).replace('.', '_')}"
    logger.info(
        "Merging: method=%s density=%.2f weights=[%.2f, %.2f]",
        method, density, weight_a, weight_b,
    )

    kwargs: Dict[str, Any] = {
        "adapters": ["sft", "ssl"],
        "weights": [weight_a, weight_b],
        "adapter_name": merge_name,
        "combination_type": method,
    }
    if method in ("ties", "dare_ties", "dare_linear"):
        kwargs["density"] = density

    model.add_weighted_adapter(**kwargs)
    model.set_adapter(merge_name)

    # Delete non-merged adapters so save_pretrained only writes the merged one
    model.delete_adapter("sft")
    model.delete_adapter("ssl")

    # Save merged adapter — PEFT puts files in adapter_name/ subdirectory,
    # so we save to output_path directly then move files to lora_adapter/
    import shutil

    tmp_save_dir = Path(output_path) / "_merge_tmp"
    tmp_save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(tmp_save_dir))

    # Move the adapter subdirectory to lora_adapter/
    adapter_subdir = tmp_save_dir / merge_name
    final_dir = Path(output_path) / "lora_adapter"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.move(str(adapter_subdir), str(final_dir))
    shutil.rmtree(tmp_save_dir, ignore_errors=True)
    logger.info("Merged adapter saved to %s", final_dir)

    # Cleanup
    del model
    gc.collect()
    try:
        import torch as _torch

        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception:
        pass

    return str(final_dir)


def run_eval(
    config_path: str,
    output_dir: str,
    test_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """Run evaluation using the merged adapter.

    Overrides the config's output_dir so that ``BaselineTrainer.evaluate()``
    picks up the merged adapter from ``output_dir/lora_adapter/``.
    """
    import yaml

    from cora.baseline.config import BaselineConfig
    from cora.baseline.finetune import BaselineTrainer

    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    # Override output_dir to point to merged adapter location
    cfg_dict["output_dir"] = str(Path(output_dir).parent)
    cfg_dict["model"]["name"] = Path(output_dir).name

    config = BaselineConfig(**cfg_dict)

    test_csv_path = test_csv or config.data_test_csv
    if not test_csv_path:
        raise ValueError("No test CSV. Use --test-csv or set data_test_csv in config.")

    trainer = BaselineTrainer(config)
    eval_output_dir = str(Path(output_dir) / "eval")
    metrics = trainer.evaluate(test_csv=test_csv_path, output_dir=eval_output_dir)

    logger.info("Eval metrics: %s", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}, indent=2))
    return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Merge two LoRA adapters via PEFT and evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Baseline YAML config (for model ID & eval params)")
    parser.add_argument("--adapter-a", type=str, required=True, help="Path to first LoRA adapter (e.g. SFT-only)")
    parser.add_argument("--adapter-b", type=str, required=True, help="Path to second LoRA adapter (e.g. DenseCL)")
    parser.add_argument("--method", type=str, default="ties", choices=["ties", "dare_ties", "dare_linear", "linear", "cat", "svd"], help="Merge method (default: ties)")
    parser.add_argument("--density", type=float, nargs="+", default=[0.5], help="Density for TIES/DARE (default: 0.5). Multiple values = sweep.")
    parser.add_argument("--weight-a", type=float, default=1.0, help="Weight for adapter A")
    parser.add_argument("--weight-b", type=float, default=1.0, help="Weight for adapter B")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for merged adapter + eval")
    parser.add_argument("--test-csv", type=str, default=None, help="Override test CSV path")
    parser.add_argument("--eval-only", action="store_true", help="Skip merging, only run eval (adapter must exist)")
    parser.add_argument("--merge-only", action="store_true", help="Skip eval, only run merge")

    args = parser.parse_args()

    densities = args.density
    all_results: Dict[str, Dict[str, Any]] = {}

    for density in densities:
        # Build output path for this density
        if len(densities) > 1:
            run_dir = str(Path(args.output_dir) / f"d{density:.2f}")
        else:
            run_dir = args.output_dir

        logger.info("=" * 60)
        logger.info("Method=%s  Density=%.2f  Output=%s", args.method, density, run_dir)
        logger.info("=" * 60)

        # Merge
        if not args.eval_only:
            merge_adapters(
                config_path=args.config,
                adapter_a_path=args.adapter_a,
                adapter_b_path=args.adapter_b,
                output_path=run_dir,
                method=args.method,
                density=density,
                weight_a=args.weight_a,
                weight_b=args.weight_b,
            )

        # Eval
        if not args.merge_only:
            metrics = run_eval(
                config_path=args.config,
                output_dir=run_dir,
                test_csv=args.test_csv,
            )
            all_results[f"{args.method}_d{density:.2f}"] = metrics

    # Print summary
    if all_results:
        print("\n" + "=" * 60)
        print("MERGE RESULTS SUMMARY")
        print("=" * 60)
        for name, m in all_results.items():
            cider = m.get("cider", m.get("CIDEr", "N/A"))
            bleu = m.get("bleu4", m.get("Bleu_4", "N/A"))
            print(f"  {name}: CIDEr={cider}  BLEU4={bleu}")
        print("=" * 60)


if __name__ == "__main__":
    main()
