from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from cora.config.schema import CORAConfig, StageConfig

logger = logging.getLogger(__name__)


class ConfigManager:

    @staticmethod
    def load(yaml_path: Union[str, Path]) -> CORAConfig:
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raw = {}

        return CORAConfig(**raw)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> CORAConfig:
        return cls.load(yaml_path)

    @staticmethod
    def save(config: CORAConfig, yaml_path: Union[str, Path]) -> None:
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        data = config.model_dump(exclude_none=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, indent=2)
        logger.info("Config saved to %s", yaml_path)

    @staticmethod
    def merge_cli_overrides(config: CORAConfig, overrides: Dict[str, Any]) -> CORAConfig:
        data = config.model_dump()
        for dotted_key, value in overrides.items():
            keys = dotted_key.split(".")
            target = data
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            target[keys[-1]] = value
        return CORAConfig(**data)

    @staticmethod
    def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
        parser.add_argument("--stage", type=str, default=None, help="Run specific stage only")
        parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint (auto or path)")
        return parser

    @staticmethod
    def get_stage_config(config: CORAConfig, stage: str) -> StageConfig:
        return config.get_stage_config(stage)

    @staticmethod
    def auto_detect_config(checkpoint_path: Union[str, Path]) -> Optional[CORAConfig]:
        checkpoint_path = Path(checkpoint_path)
        search_dir = checkpoint_path.parent if checkpoint_path.is_file() else checkpoint_path

        candidates = [
            search_dir / "config.yaml",
            search_dir / "config.yml",
            search_dir.parent / "config.yaml",
        ]

        for p in candidates:
            if p.exists():
                try:
                    return ConfigManager.load(p)
                except Exception as exc:
                    logger.warning("Failed to load config from %s: %s", p, exc)
        return None
