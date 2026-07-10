# CORA Model Zoo

All entries are LoRA adapters and require the listed upstream base model. The links become the canonical
artifact locations for the `v2.0.0-eccv2026` release.

| ID | Base model | Method | Seed-42 CIDEr | Adapter |
|---|---|---|---:|---|
| `internvl3-2b-densecl` | `OpenGVLab/InternVL3-2B-hf` | DenseCL 50% | 0.3613 | [HF](https://huggingface.co/wooseungw/cora-internvl3-2b-densecl) |
| `internvl3-2b-vicreg` | `OpenGVLab/InternVL3-2B-hf` | VICReg 50% | 0.3631 | [HF](https://huggingface.co/wooseungw/cora-internvl3-2b-vicreg) |
| `qwen2.5-vl-3b-densecl` | `Qwen/Qwen2.5-VL-3B-Instruct` | DenseCL 50% | 0.3399 | [HF](https://huggingface.co/wooseungw/cora-qwen2.5-vl-3b-densecl) |
| `qwen2.5-vl-3b-vicreg` | `Qwen/Qwen2.5-VL-3B-Instruct` | VICReg 50% | 0.3428 | [HF](https://huggingface.co/wooseungw/cora-qwen2.5-vl-3b-vicreg) |
| `gemma3-4b-densecl` | `google/gemma-3-4b-it` | DenseCL 50% | 0.3422 | [HF](https://huggingface.co/wooseungw/cora-gemma3-4b-densecl) |
| `gemma3-4b-vicreg` | `google/gemma-3-4b-it` | VICReg 50% | 0.3502 | [HF](https://huggingface.co/wooseungw/cora-gemma3-4b-vicreg) |

Gemma access requires accepting Google's terms on the base-model page. Repository MIT licensing applies to
CORA source code and does not replace upstream model or dataset terms. Machine-readable mappings are in
`release/model_registry.json`.
