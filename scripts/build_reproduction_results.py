#!/usr/bin/env python3
"""Aggregate the three camera-ready seeds into checked-in reference results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

SEEDS = (42, 1234, 7)
RUNS = {
    "internvl3-2b-native": ("InternVL3-2B", "Native", "internvl_native", "trackA_internvl35-2b_native"),
    "internvl3-2b-densecl": ("InternVL3-2B", "CORA DenseCL 50%", "internvl_densecl50", "trackB_internvl35-2b_densecl50"),
    "internvl3-2b-vicreg": ("InternVL3-2B", "CORA VICReg 50%", "internvl_vicreg50", "trackB_internvl35-2b_vicreg50"),
    "qwen2.5-vl-3b-native": ("Qwen2.5-VL-3B", "Native", "qwen_native", "trackA_qwen25-3b_native"),
    "qwen2.5-vl-3b-densecl": ("Qwen2.5-VL-3B", "CORA DenseCL 50%", "qwen_densecl50", "trackB_qwen25-3b_densecl50"),
    "qwen2.5-vl-3b-vicreg": ("Qwen2.5-VL-3B", "CORA VICReg 50%", "qwen_vicreg50", "trackB_qwen25-3b_vicreg50"),
    "gemma3-4b-native": ("Gemma3-4B", "Native", "gemma_native", "trackA_gemma3-4b_native"),
    "gemma3-4b-densecl": ("Gemma3-4B", "CORA DenseCL 50%", "gemma_densecl50", "trackB_gemma3-4b_densecl50"),
    "gemma3-4b-vicreg": ("Gemma3-4B", "CORA VICReg 50%", "gemma_vicreg50", "trackB_gemma3-4b_vicreg50"),
}
METRICS = ("bleu_4", "meteor", "rouge_l", "cider", "spice")


def read_metrics(path: Path) -> dict[str, float]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return {metric: float(data[metric]) for metric in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for row_id, (backbone, method, group, run_name) in RUNS.items():
        seed_values: dict[int, dict[str, float]] = {}
        for seed in SEEDS:
            path = args.runs_root / group / f"seed_{seed}" / run_name / "eval" / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            seed_values[seed] = read_metrics(path)
        row: dict[str, Any] = {"id": row_id, "backbone": backbone, "method": method, "seeds": seed_values}
        for metric in METRICS:
            values = [seed_values[seed][metric] for seed in SEEDS]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.stdev(values)
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "seeds": list(SEEDS), "rows": rows}
    (args.output_dir / "table1_multiseed.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fields = ["id", "backbone", "method"] + [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")]
    with (args.output_dir / "table1_multiseed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
