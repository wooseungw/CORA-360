# CORA: Overlap-Consistent View Decomposition for Adapting Vision–Language Models to 360° Panoramas

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

Official implementation of **CORA** (Consistent Overlap Representation Adaptation), a lightweight,
model-agnostic framework that adapts off-the-shelf Vision–Language Models (VLMs) to 360°
panoramic inputs **without modifying the backbone architecture**.

> **Seungwoo Woo, Daewon Jung, Sekyoung Youm** — Dongguk University, Seoul, South Korea

Applied to three architecturally diverse VLMs on QuIC-360, CORA improves CIDEr by up to
**+0.0204** over the Native baseline (0.3405 → 0.3609) while training only LoRA adapters
(**~0.6%** of backbone parameters). A key finding: *perspective view decomposition alone
accounts for 95% of the total gain.*

---

## 🔭 Overview

Off-the-shelf VLMs are trained on perspective (rectilinear) images and struggle with
equirectangular (ERP) panoramas. CORA addresses this with three **parameter-free** components;
only LoRA adapters are trained while the backbone stays frozen.

| Component | Role |
|---|---|
| **AnyRes-E2P** | Decomposes the ERP panorama into a closed loop of overlapping, low-distortion perspective views via gnomonic projection (9 views = 8 tiles @ 45° stride + 1 global ERP stream). |
| **PanoRoPE** | Deterministic, parameter-free positional remapping that encodes the circular panoramic topology (width-axis for M-RoPE; 1-D shift for 1-D RoPE). |
| **Overlap-consistency loss** | Self-supervised feature alignment at view boundaries (VICReg-pairwise or DenseCL/InfoNCE). |

Evaluated on **InternVL3.5-2B**, **Qwen2.5-VL-3B**, and **Gemma3-4B** using
[QuIC-360](https://aclanthology.org/2023.emnlp-main.1093/) (query-based panoramic captioning).

## 📊 Main results (QuIC-360 test, Table 1)

CORA = AnyRes-E2P + PanoRoPE + overlap loss; all rows use LoRA for 1 epoch. *Judge* = LLM-judge score (0–100).

| Model | Method | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE | Judge |
|---|---|---|---|---|---|---|---|
| **InternVL3.5-2B** | Resize | 0.0403 | 0.1096 | 0.2402 | 0.3054 | 0.1566 | 67.6 |
| | Native | 0.0443 | 0.1111 | 0.2462 | 0.3405 | 0.1661 | 69.2 |
| | CORA DenseCL 50% | 0.0457 | 0.1137 | 0.2492 | 0.3603 | 0.1720 | **70.6** |
| | CORA VICReg 50% | 0.0456 | 0.1137 | 0.2491 | **0.3609** | 0.1718 | 70.2 |
| **Qwen2.5-VL-3B** | Resize | 0.0382 | 0.1113 | 0.2334 | 0.2809 | 0.1435 | 65.3 |
| | Native | 0.0434 | 0.1125 | 0.2427 | 0.3306 | 0.1548 | 67.9 |
| | CORA DenseCL 50% | 0.0426 | 0.1135 | 0.2448 | 0.3423 | 0.1613 | 68.0 |
| | CORA VICReg 50% | 0.0426 | 0.1137 | 0.2446 | **0.3425** | 0.1627 | **68.3** |
| **Gemma3-4B** | Resize | 0.0421 | 0.1085 | 0.2449 | 0.3383 | 0.1640 | 68.8 |
| | Native | 0.0420 | 0.1081 | 0.2453 | 0.3363 | 0.1636 | 68.6 |
| | CORA DenseCL 50% | 0.0437 | 0.1162 | 0.2415 | 0.3362 | 0.1685 | 69.3 |
| | CORA VICReg 50% | 0.0440 | 0.1180 | 0.2410 | 0.3357 | 0.1700 | **71.8** |

DenseCL and VICReg-pairwise produce near-identical CIDEr (formulation-agnostic). Gains correlate
with the vision encoder's gradient accessibility; see the paper for the full ablation and analysis.

---

## ⚙️ Installation

```bash
# 1) Create the environment
conda create -n pano python=3.12 -y
conda activate pano

# 2) Install the package (editable)
pip install -e ".[dev]"

# 3) (optional) Caption metrics: BLEU / METEOR / ROUGE-L / CIDEr / SPICE
bash install_eval_metrics.sh   # SPICE needs a JDK (apt install default-jdk)
```

Backbones are pulled from the HuggingFace Hub on first use
(`OpenGVLab/InternVL3-2B-hf`, `Qwen/Qwen2.5-VL-3B-Instruct`, `google/gemma-3-4b-it`).
Gemma3 is a gated model — request access and run `huggingface-cli login`.

📖 Full environment, model, and dataset setup: **[docs/SETUP.md](docs/SETUP.md)**.

## 📁 Dataset

CORA is trained and evaluated on **QuIC-360**. Provide CSVs with columns `url, instruction, response`
and point each config at them (default: `runs/baseline/_shared_data/{train,test}.csv`).
QuIC-360 images are hosted on Flickr; see [docs/SETUP.md](docs/SETUP.md) for the download workflow.

```
url,instruction,response
/path/to/pano.jpg,What do you see?,"A wide panoramic scene ..."
```

## 🚀 Usage

Training and evaluation are fully config-driven (`configs/baseline/*.yaml`).

```bash
conda activate pano
export CUDA_VISIBLE_DEVICES=0

# Train (LoRA, 1 epoch) — writes to runs/baseline/<experiment_name>/
python scripts/baseline_finetune.py \
    --config configs/baseline/panoadapt_vicreg_pairwise_internvl35_2b.yaml

# Evaluate (BLEU/METEOR/ROUGE-L/CIDEr/SPICE)
python scripts/baseline_eval.py \
    --config configs/baseline/panoadapt_vicreg_pairwise_internvl35_2b.yaml

# LLM-as-a-judge (multimodal GPT-based scoring; needs OPENAI_API_KEY)
python scripts/llm_judge_eval.py --help
```

### Reproducing Table 1

Each config's `experiment_name` maps to its output directory under `runs/baseline/`.

| Paper row | InternVL3.5-2B | Qwen2.5-VL-3B | Gemma3-4B |
|---|---|---|---|
| **Native** | `native_internvl35_2b.yaml` | `native_qwen25_3b.yaml` | `native_gemma3_4b.yaml` |
| **CORA DenseCL 50%** | `panoadapt_internvl35_2b.yaml` | `panoadapt_pe_densecl_qwen25_3b.yaml` | `panoadapt_gemma3_4b.yaml` |
| **CORA VICReg 50%** | `panoadapt_vicreg_pairwise_internvl35_2b.yaml` | `panoadapt_vicreg_pairwise_qwen25_3b.yaml` | `panoadapt_vicreg_pairwise_gemma3_4b.yaml` |

Ablations: view construction (Table 2a) → `ablation_internvl35_2b_{cubemap_noloss,anyrese2p_noloss}.yaml`,
`cubemap_qwen25_3b.yaml`, `pinhole_qwen25_3b.yaml`, `anyres_e2p_qwen25_3b.yaml`;
component-wise (Table 2b) → `ablation_internvl35_2b_anyrese2p_pe_only.yaml`;
FoV / view-count sweep (Table S5) → `e4a–e4d_internvl35-2b_*.yaml`.

## 🗂️ Repository structure

```
panollava/
├── src/cora/              # CORA package (pip install -e .)
│   ├── baseline/          #   LoRA finetune + eval pipeline (panoadapt)
│   ├── model/             #   AnyRes-E2P, PanoRoPE, projectors, vision encoder
│   ├── processors/        #   ERP → perspective view construction
│   ├── training/          #   trainer, losses (VICReg / DenseCL overlap), callbacks
│   └── config/, evaluation/, inference/
├── configs/baseline/      # Experiment configs (paper experiments)
├── scripts/               # baseline_finetune.py, baseline_eval.py, llm_judge_eval.py
├── docs/                  # SETUP.md, ARCHITECTURE.md, qualitative examples
└── tests/
```

## 📝 Citation

If you find CORA useful, please cite:

```bibtex
@inproceedings{woo2026cora,
  title     = {Overlap-Consistent View Decomposition for Adapting
               Vision--Language Models to 360{\textdegree} Panoramas},
  author    = {Woo, Seungwoo and Jung, Daewon and Youm, Sekyoung},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## 🙏 Acknowledgments

Built on [Transformers](https://github.com/huggingface/transformers),
[PEFT](https://github.com/huggingface/peft), and
[PyTorch Lightning](https://github.com/Lightning-AI/lightning). Backbones:
[InternVL](https://github.com/OpenGVLab/InternVL), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL),
and [Gemma](https://ai.google.dev/gemma). Benchmark: [QuIC-360](https://aclanthology.org/2023.emnlp-main.1093/).

## 📄 License

Released under the [MIT License](LICENSE).
