#!/usr/bin/env python3
"""Baseline VLM evaluation — generate predictions on test set and compute metrics."""
import argparse
import json

import yaml


def main():
    parser = argparse.ArgumentParser(description="Baseline VLM Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to baseline YAML config")
    parser.add_argument("--test-csv", type=str, default=None, help="Override test CSV path")
    parser.add_argument("--train-output-dir", type=str, default=None,
                        help="Override training output directory used to locate the LoRA adapter")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    from cora.baseline.config import BaselineConfig
    from cora.baseline.finetune import BaselineTrainer

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    if args.train_output_dir:
        cfg_dict["output_dir"] = args.train_output_dir
    config = BaselineConfig(**cfg_dict)

    test_csv = args.test_csv or config.data_test_csv
    if not test_csv:
        raise ValueError("No test CSV provided. Use --test-csv or set data_test_csv in config.")

    trainer = BaselineTrainer(config)
    metrics = trainer.evaluate(test_csv=test_csv, output_dir=args.output_dir)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
