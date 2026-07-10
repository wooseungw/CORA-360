# CORA Model Zoo

All entries are LoRA adapters and require the listed upstream base model. The links become the canonical
artifact locations for the `v2.0.0-eccv2026` release.

| ID | Base model | Method | 🤗 Hub | Seed-42 CIDEr | Release archive |
|---|---|---|---|---:|---|
| `internvl3-2b-densecl` | `OpenGVLab/InternVL3-2B-hf` | DenseCL 50% | [wfwefw/cora-internvl3-2b-densecl](https://huggingface.co/wfwefw/cora-internvl3-2b-densecl) | 0.3613 | [tar.gz](https://github.com/wooseungw/CORA-360/releases/download/v2.0.0-eccv2026/cora-internvl3-2b-densecl.tar.gz) |
| `internvl3-2b-vicreg` | `OpenGVLab/InternVL3-2B-hf` | VICReg 50% | [wfwefw/cora-internvl3-2b-vicreg](https://huggingface.co/wfwefw/cora-internvl3-2b-vicreg) | 0.3631 | [tar.gz](https://github.com/wooseungw/CORA-360/releases/download/v2.0.0-eccv2026/cora-internvl3-2b-vicreg.tar.gz) |
| `qwen2.5-vl-3b-densecl` | `Qwen/Qwen2.5-VL-3B-Instruct` | DenseCL 50% | [wfwefw/cora-qwen2.5-vl-3b-densecl](https://huggingface.co/wfwefw/cora-qwen2.5-vl-3b-densecl) | 0.3399 | [tar.gz](https://github.com/wooseungw/CORA-360/releases/download/v2.0.0-eccv2026/cora-qwen2.5-vl-3b-densecl.tar.gz) |
| `qwen2.5-vl-3b-vicreg` | `Qwen/Qwen2.5-VL-3B-Instruct` | VICReg 50% | [wfwefw/cora-qwen2.5-vl-3b-vicreg](https://huggingface.co/wfwefw/cora-qwen2.5-vl-3b-vicreg) | 0.3428 | [tar.gz](https://github.com/wooseungw/CORA-360/releases/download/v2.0.0-eccv2026/cora-qwen2.5-vl-3b-vicreg.tar.gz) |
| `gemma3-4b-densecl` | `google/gemma-3-4b-it` | DenseCL 50% | [wfwefw/cora-gemma3-4b-densecl](https://huggingface.co/wfwefw/cora-gemma3-4b-densecl) | 0.3422 | [tar.gz](https://github.com/wooseungw/CORA-360/releases/download/v2.0.0-eccv2026/cora-gemma3-4b-densecl.tar.gz) |
| `gemma3-4b-vicreg` | `google/gemma-3-4b-it` | VICReg 50% | [wfwefw/cora-gemma3-4b-vicreg](https://huggingface.co/wfwefw/cora-gemma3-4b-vicreg) | 0.3502 | [tar.gz](https://github.com/wooseungw/CORA-360/releases/download/v2.0.0-eccv2026/cora-gemma3-4b-vicreg.tar.gz) |

Every adapter is mirrored on the 🤗 Hub (linked above) and attached to the
[`v2.0.0-eccv2026`](https://github.com/wooseungw/CORA-360/releases/tag/v2.0.0-eccv2026) GitHub Release as a
`tar.gz`; both carry the same `safetensors` weights and model card. Machine-readable base-model revisions and
run provenance are in `release/model_registry.json`.

Gemma access requires accepting Google's terms on the base-model page. Repository MIT licensing applies to
CORA source code and does not replace upstream model or dataset terms.
