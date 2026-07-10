# CORA: Overlap-Consistent View Decomposition for Adapting Vision–Language Models to 360° Panoramas

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CI](https://github.com/wooseungw/CORA-360/actions/workflows/ci.yml/badge.svg)](https://github.com/wooseungw/CORA-360/actions/workflows/ci.yml)
[![Models on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Models-CORA--360-yellow)](https://huggingface.co/wfwefw)

Official implementation of **CORA** (Consistent Overlap Representation Adaptation), a lightweight,
model-agnostic framework that adapts off-the-shelf Vision–Language Models (VLMs) to 360°
panoramic inputs **without modifying the backbone architecture**.

> **Seungwoo Woo, Daewon Jung, Sekyoung Youm** — Dongguk University, Seoul, South Korea

Applied to three architecturally diverse VLMs on QuIC-360, CORA improves three-seed mean CIDEr by up to
**+0.0166** over the Native baseline (0.3444 → 0.3610) while training only LoRA adapters
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

Evaluated on **InternVL3-2B**, **Qwen2.5-VL-3B**, and **Gemma3-4B** using
[QuIC-360](https://aclanthology.org/2023.findings-emnlp.463/) (query-based panoramic captioning).

## 📊 Main results (QuIC-360 test, Table 1)

CORA = AnyRes-E2P + PanoRoPE + overlap loss; all rows use LoRA for 1 epoch. Values are mean ± sample
standard deviation over seeds 42, 1234, and 7. Machine-readable per-seed values are in
[`results/table1_multiseed.json`](results/table1_multiseed.json).

| Model | Method | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---|---:|---:|---:|---:|---:|
| **InternVL3-2B** | Native | .0437±.0011 | .1114±.0014 | .2462±.0017 | .3444±.0096 | .1670±.0017 |
| | CORA DenseCL 50% | .0445±.0007 | .1135±.0006 | .2490±.0009 | .3602±.0016 | .1716±.0011 |
| | CORA VICReg 50% | .0450±.0008 | .1137±.0006 | .2491±.0003 | **.3610±.0047** | .1722±.0011 |
| **Qwen2.5-VL-3B** | Native | .0422±.0004 | .1122±.0010 | .2409±.0008 | .3257±.0028 | .1547±.0012 |
| | CORA DenseCL 50% | .0439±.0002 | .1139±.0002 | .2437±.0008 | .3389±.0010 | .1601±.0005 |
| | CORA VICReg 50% | .0427±.0002 | .1142±.0008 | .2439±.0010 | **.3410±.0016** | .1612±.0010 |
| **Gemma3-4B** | Native | .0422±.0001 | .1088±.0002 | .2441±.0011 | .3369±.0048 | .1647±.0014 |
| | CORA DenseCL 50% | .0444±.0002 | .1154±.0007 | .2441±.0013 | .3417±.0011 | .1662±.0007 |
| | CORA VICReg 50% | .0441±.0006 | .1140±.0014 | .2455±.0014 | **.3484±.0031** | .1669±.0019 |

DenseCL and VICReg-pairwise produce near-identical CIDEr (formulation-agnostic). Gains correlate
with the vision encoder's gradient accessibility; see the paper for the full ablation and analysis.

---

## ⚙️ Installation

```bash
# 1) Create the environment
conda env create -f environment.yml
conda activate cora-eccv2026
```

Backbones are pulled from the HuggingFace Hub on first use
(`OpenGVLab/InternVL3-2B-hf`, `Qwen/Qwen2.5-VL-3B-Instruct`, `google/gemma-3-4b-it`).
Gemma3 is a gated model — request access and run `huggingface-cli login`.

📖 Full environment, model, and dataset setup: **[docs/SETUP.md](docs/SETUP.md)**.

## 📁 Dataset

CORA is trained and evaluated on **QuIC-360**. Provide CSVs with columns `url, instruction, response`.
The exact 7,929/5,349 split and row order can be verified without redistributing the annotations:

```bash
./reproduce.sh data-check /path/to/train.csv /path/to/test.csv
```

```
url,instruction,response
/path/to/pano.jpg,What do you see?,"A wide panoramic scene ..."
```

## 🚀 Usage

Training and evaluation are fully config-driven (`configs/baseline/*.yaml`).

All six trained LoRA adapters are on the [🤗 Hub](https://huggingface.co/wfwefw) and mirrored on the
GitHub Release. For per-model links, checksums, and one-command evaluation, see
[MODEL_ZOO.md](MODEL_ZOO.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

```bash
conda activate cora-eccv2026
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

| Paper row | InternVL3-2B | Qwen2.5-VL-3B | Gemma3-4B |
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
CORA-360/
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
and [Gemma](https://ai.google.dev/gemma). Benchmark: [QuIC-360](https://aclanthology.org/2023.findings-emnlp.463/).

## 📄 License

The CORA source code is released under the [MIT License](LICENSE). Upstream model and dataset licenses still
apply; in particular, Gemma adapter users must accept the Gemma terms before accessing the base model.
