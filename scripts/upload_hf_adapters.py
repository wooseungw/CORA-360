#!/usr/bin/env python3
"""Upload prepared CORA adapter directories to their Hugging Face repositories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("release/model_registry.json"))
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    api = HfApi()
    api.whoami()
    for model in registry["models"]:
        repo_id = model["hf_repo"]
        folder = args.artifact_root / model["id"]
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=folder,
            commit_message="Publish CORA ECCV 2026 seed-42 adapter",
        )
        print(f"Uploaded https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
