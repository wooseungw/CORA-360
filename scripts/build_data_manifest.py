#!/usr/bin/env python3
"""Create a path-independent, privacy-preserving manifest for QuIC-360 splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def digest(value: str) -> str:
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()


def rows_for(split: str, path: Path) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            image = row.get("url") or row.get("image") or ""
            query = row.get("instruction") if "instruction" in row else row.get("query", "")
            response = row.get("response") if "response" in row else row.get("annotation", "")
            if not image or query is None or response is None:
                raise ValueError(f"Missing required column in {path}:{index + 2}")
            output.append(
                {
                    "split": split,
                    "row_index": str(index),
                    "image_filename": Path(image).name,
                    "query_sha256": digest(query),
                    "response_sha256": digest(response),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/manifests/quic360_split_manifest.csv"))
    args = parser.parse_args()
    rows = rows_for("train", args.train) + rows_for("test", args.test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
