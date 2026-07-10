#!/usr/bin/env python3
"""Compare an evaluation metrics file with the released seed-42 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRICS = ("bleu_4", "meteor", "rouge_l", "cider", "spice")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("results/table1_multiseed.json"))
    parser.add_argument("--tolerance", type=float, default=0.002)
    args = parser.parse_args()
    actual: dict[str, Any] = json.loads(args.actual.read_text(encoding="utf-8"))
    reference: dict[str, Any] = json.loads(args.reference.read_text(encoding="utf-8"))
    row = next((item for item in reference["rows"] if item["id"] == args.model), None)
    if row is None:
        raise SystemExit(f"Unknown model id: {args.model}")
    expected = row["seeds"]["42"]
    failures = []
    for metric in METRICS:
        delta = abs(float(actual[metric]) - float(expected[metric]))
        print(f"{metric}: actual={actual[metric]:.6f} reference={expected[metric]:.6f} delta={delta:.6f}")
        if delta > args.tolerance:
            failures.append(metric)
    if failures:
        raise SystemExit(f"Metrics outside tolerance {args.tolerance}: {', '.join(failures)}")


if __name__ == "__main__":
    main()
