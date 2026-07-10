# CORA Model Zoo

All entries are LoRA adapters and require the listed upstream base model. The links become the canonical
artifact locations for the `v2.0.0-eccv2026` release.

| ID | Base model | Method | Seed-42 CIDEr | Release archive |
|---|---|---|---:|---|
| `internvl3-2b-densecl` | `OpenGVLab/InternVL3-2B-hf` | DenseCL 50% | 0.3613 | [tar.gz](https://github.com/wooseungw/CORA-360/releases/download/v2.0.0-eccv2026/cora-internvl3-2b-densecl.tar.gz) |
| `internvl3-2b-vicreg` | `OpenGVLab/InternVL3-2B-hf` | VICReg 50% | 0.3631 | [tar.gz](https://github.com/wooseungw/CORA-360/releases/download/v2.0.0-eccv2026/cora-internvl3-2b-vicreg.tar.gz) |
| `qwen2.5-vl-3b-densecl` | `Qwen/Qwen2.5-VL-3B-Instruct` | DenseCL 50% | 0.3399 | [tar.gz](https://github.com/wooseungw/CORA-360/releases/download/v2.0.0-eccv2026/cora-qwen2.5-vl-3b-densecl.tar.gz) |
| `qwen2.5-vl-3b-vicreg` | `Qwen/Qwen2.5-VL-3B-Instruct` | VICReg 50% | 0.3428 | [tar.gz](https://github.com/wooseungw/CORA-360/releases/download/v2.0.0-eccv2026/cora-qwen2.5-vl-3b-vicreg.tar.gz) |
| `gemma3-4b-densecl` | `google/gemma-3-4b-it` | DenseCL 50% | 0.3422 | [tar.gz](https://github.com/wooseungw/CORA-360/releases/download/v2.0.0-eccv2026/cora-gemma3-4b-densecl.tar.gz) |
| `gemma3-4b-vicreg` | `google/gemma-3-4b-it` | VICReg 50% | 0.3502 | [tar.gz](https://github.com/wooseungw/CORA-360/releases/download/v2.0.0-eccv2026/cora-gemma3-4b-vicreg.tar.gz) |

Gemma access requires accepting Google's terms on the base-model page. Repository MIT licensing applies to
CORA source code and does not replace upstream model or dataset terms. Each archive contains a Hugging Face
model card and can also be mirrored to the Hub using the machine-readable mappings in `release/model_registry.json`.
