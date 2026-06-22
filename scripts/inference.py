#!/usr/bin/env python3
"""CORA single-image inference."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="CORA Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--question", type=str, required=True, help="Question about the image")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--config", type=str, default=None, help="Optional config file")
    args = parser.parse_args()

    from cora.inference.generator import PanoramaGenerator

    print(f"Loading model from: {args.checkpoint}")
    print(f"Image: {args.image}")
    print(f"Question: {args.question}")


if __name__ == "__main__":
    main()
