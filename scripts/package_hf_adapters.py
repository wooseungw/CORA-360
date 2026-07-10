#!/usr/bin/env python3
"""Build self-contained Hugging Face upload directories for released adapters."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_card(model: dict[str, Any], metrics: dict[str, Any]) -> str:
    base_model = model["base_model"]
    license_name = "gemma" if base_model.startswith("google/gemma") else "other"
    license_note = (
        "This adapter is subject to the Gemma Terms of Use in addition to the CORA source-code license."
        if license_name == "gemma"
        else "Use of the base model remains subject to its upstream license."
    )
    return f"""---
pipeline_tag: image-text-to-text
library_name: peft
base_model: {base_model}
datasets:
  - Silviase/QuIC-360
license: {license_name}
tags:
  - cora
  - panorama
  - 360-degree
  - lora
  - vision-language
---

# {model['display_name']}

Seed-42 LoRA adapter for the ECCV 2026 CORA release. It must be used with
`{base_model}` at revision `{model['base_revision']}` and the matching CORA
config included as `cora_config.yaml`.

## Evaluation

QuIC-360 test set ({metrics['samples']} query-caption pairs):

| BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---:|---:|---:|---:|---:|
| {metrics['bleu_4']:.4f} | {metrics['meteor']:.4f} | {metrics['rouge_l']:.4f} | {metrics['cider']:.4f} | {metrics['spice']:.4f} |

Three-seed aggregate results and exact split hashes are maintained in the
[CORA-360 repository](https://github.com/wooseungw/CORA-360).

## Usage

```bash
git clone --branch v2.0.0-eccv2026 https://github.com/wooseungw/CORA-360.git
cd CORA-360
./reproduce.sh evaluate {model['id']} /path/to/test.csv
```

## Limitations and license

CORA was evaluated on English query-focused captions from QuIC-360. Performance
outside that domain, on non-ERP imagery, or in safety-critical settings is not established.
{license_note}

## Citation

```bibtex
@inproceedings{{woo2026cora,
  title={{Overlap-Consistent View Decomposition for Adapting Vision--Language Models to 360-Degree Panoramas}},
  author={{Woo, Seungwoo and Jung, Daewon and Youm, Sekyoung}},
  booktitle={{European Conference on Computer Vision}},
  year={{2026}}
}}
```
"""


def write_sanitized_predictions(source: Path, destination: Path) -> None:
    """Copy predictions without publishing workstation-specific absolute paths."""
    with source.open(encoding="utf-8-sig", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing prediction header: {source}")
        fieldnames = ["image_filename" if name == "image_path" else name for name in reader.fieldnames]
        with destination.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in reader:
                output = {
                    ("image_filename" if key == "image_path" else key): value for key, value in row.items()
                }
                output["image_filename"] = Path(output["image_filename"]).name
                writer.writerow(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads((args.repo_root / "release/model_registry.json").read_text(encoding="utf-8"))
    for model in registry["models"]:
        source = args.runs_root / model["run"]
        adapter = source / "lora_adapter"
        metrics_path = source / "eval/metrics.json"
        if not adapter.is_dir() or not metrics_path.is_file():
            raise FileNotFoundError(source)
        destination = args.output_root / model["id"]
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        for name in ("adapter_model.safetensors", "adapter_config.json", "panoadapt_yaw_rope.safetensors"):
            candidate = adapter / name
            if candidate.is_file():
                shutil.copy2(candidate, destination / name)
        adapter_config_path = destination / "adapter_config.json"
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        adapter_config["revision"] = model["base_revision"]
        adapter_config_path.write_text(json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8")
        shutil.copy2(args.repo_root / model["config"], destination / "cora_config.yaml")
        shutil.copy2(metrics_path, destination / "metrics.json")
        write_sanitized_predictions(source / "eval/predictions.csv", destination / "predictions.csv")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        training_manifest = {
            "schema_version": 1,
            "code_repository": registry["code_repository"],
            "code_release": registry["release_tag"],
            "base_model": model["base_model"],
            "base_revision": model["base_revision"],
            "config": "cora_config.yaml",
            "data_manifest": "data/manifests/quic360_split_manifest.csv",
            "method": model["method"],
            "seed": model["seed"],
            "evaluation_samples": metrics["samples"],
        }
        (destination / "training_manifest.json").write_text(
            json.dumps(training_manifest, indent=2) + "\n", encoding="utf-8"
        )
        (destination / "README.md").write_text(model_card(model, metrics), encoding="utf-8")
        files = sorted(path for path in destination.iterdir() if path.is_file() and path.name != "SHA256SUMS")
        checksums = "".join(f"{sha256(path)}  {path.name}\n" for path in files)
        (destination / "SHA256SUMS").write_text(checksums, encoding="utf-8")
        print(f"Prepared {model['hf_repo']} at {destination}")


if __name__ == "__main__":
    main()
