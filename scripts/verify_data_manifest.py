#!/usr/bin/env python3
"""Verify local QuIC-360 CSVs against the released split manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from build_data_manifest import rows_for


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/quic360_split_manifest.csv"))
    args = parser.parse_args()
    actual = rows_for("train", args.train) + rows_for("test", args.test)
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        expected = list(csv.DictReader(handle))
    if actual != expected:
        raise SystemExit("QuIC-360 split does not match the ECCV 2026 release manifest")
    print(f"Verified {len(actual)} examples ({len(rows_for('train', args.train))} train, {len(rows_for('test', args.test))} test)")


if __name__ == "__main__":
    main()
