#!/usr/bin/env python3
"""CORA single-image inference — load a trained LoRA adapter and caption a panorama.

Example:
    python scripts/inference.py \
        --config configs/baseline/panoadapt_vicreg_pairwise_internvl35_2b.yaml \
        --image path/to/panorama.jpg \
        --question "Describe the panorama image."

The adapter is resolved from the config (``<output_dir>/<model.name>/lora_adapter``)
unless ``--adapter-dir`` is given. Decoding is greedy by default; pass
``--temperature`` to sample.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="CORA single-image inference")
    parser.add_argument("--config", type=str, required=True,
                        help="Baseline YAML config used for training")
    parser.add_argument("--image", type=str, required=True, help="Path to the panorama image")
    parser.add_argument("--question", type=str, default="Describe the image.",
                        help="Prompt / question about the image")
    parser.add_argument("--adapter-dir", type=str, default=None,
                        help="LoRA adapter directory (default: <output_dir>/<model.name>/lora_adapter)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max new tokens")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (omit for deterministic greedy decoding)")
    args = parser.parse_args()

    import yaml

    from cora.baseline.config import BaselineConfig
    from cora.baseline.finetune import BaselineTrainer

    with open(args.config) as f:
        config = BaselineConfig(**yaml.safe_load(f))

    gen_cfg = {}
    if args.max_tokens:
        gen_cfg["max_new_tokens"] = args.max_tokens
    if args.temperature is not None:
        gen_cfg.update(do_sample=True, temperature=args.temperature)

    trainer = BaselineTrainer(config)
    print(f"Loading {config.model.hf_model_id} + adapter ...", flush=True)
    answer = trainer.generate_caption(
        image=args.image,
        prompt=args.question,
        adapter_dir=args.adapter_dir,
        generation_config=gen_cfg or None,
    )

    print(f"\nImage:    {args.image}")
    print(f"Question: {args.question}")
    print(f"Answer:   {answer}")


if __name__ == "__main__":
    main()
