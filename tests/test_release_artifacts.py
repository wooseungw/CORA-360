from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_model_registry_references_existing_configs() -> None:
    registry = json.loads((ROOT / "release/model_registry.json").read_text(encoding="utf-8"))
    assert len(registry["models"]) == 6
    assert len({model["id"] for model in registry["models"]}) == 6
    for model in registry["models"]:
        config_path = ROOT / model["config"]
        assert config_path.is_file()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["model"]["hf_model_id"] == model["base_model"]
        assert config["training"]["seed"] == model["seed"]


def test_reference_results_cover_three_seeds() -> None:
    results = json.loads((ROOT / "results/table1_multiseed.json").read_text(encoding="utf-8"))
    assert results["seeds"] == [42, 1234, 7]
    assert len(results["rows"]) == 9
    for row in results["rows"]:
        assert set(row["seeds"]) == {"42", "1234", "7"}
        assert 0.0 < row["cider_mean"] < 1.0


def test_split_manifest_counts_and_has_no_absolute_paths() -> None:
    manifest = ROOT / "data/manifests/quic360_split_manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert sum(row["split"] == "train" for row in rows) == 7929
    assert sum(row["split"] == "test" for row in rows) == 5349
    assert all("/" not in row["image_filename"] for row in rows)
    assert all(len(row["query_sha256"]) == 64 for row in rows)
    assert all(len(row["response_sha256"]) == 64 for row in rows)
