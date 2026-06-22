# Setup Guide — Models & Datasets

## 1. Environment

```bash
# 1) Create the conda environment
conda create -n pano python=3.12 -y
conda activate pano

# 2) Install the package (editable)
pip install -e ".[dev]"

# 3) (optional) Caption metrics: BLEU / METEOR / ROUGE-L / CIDEr / SPICE
bash install_eval_metrics.sh
```

---

## 2. Models

All models are pulled from the HuggingFace Hub automatically on the first
`from_pretrained()` call. To pre-download them, use the commands below.

### Prerequisites

```bash
pip install huggingface_hub
huggingface-cli login   # needed for gated models (e.g. Gemma 3)
```

### Models used

| Model | HuggingFace ID | Size | Role |
|-------|----------------|------|------|
| InternVL3-2B | `OpenGVLab/InternVL3-2B-hf` | ~4 GB | CORA backbone (Table 1) |
| Qwen2.5-VL-3B | `Qwen/Qwen2.5-VL-3B-Instruct` | ~6 GB | CORA backbone (Table 1) |
| Gemma 3-4B | `google/gemma-3-4b-it` | ~8 GB | CORA backbone (Table 1, gated) |
| Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | ~14 GB | larger-model study |
| Qwen2-VL-2B | `Qwen/Qwen2-VL-2B-Instruct` | ~4 GB | lightweight study |
| InternVL3-1B | `OpenGVLab/InternVL3-1B-hf` | ~2 GB | lightweight study |
| InternVL2.5-2B | `OpenGVLab/InternVL2_5-2B` | ~4 GB | additional baseline |
| InternVL2.5-4B | `OpenGVLab/InternVL2_5-4B` | ~8 GB | additional baseline |
| BLIP2-2.7B | `Salesforce/blip2-opt-2.7b` | ~6 GB | additional baseline |

### Pre-download (optional)

```bash
# The three CORA backbones (~18 GB)
huggingface-cli download OpenGVLab/InternVL3-2B-hf
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct
huggingface-cli download google/gemma-3-4b-it   # gated: request access first
```

> **Gemma 3 access**: open <https://huggingface.co/google/gemma-3-4b-it>, click
> "Access repository", then run `huggingface-cli login`.

Models are cached under `~/.cache/huggingface/hub/` by default. To change the
location: `export HF_HOME=/your/storage/path`.

---

## 3. Dataset — QuIC-360

CORA is trained and evaluated on **QuIC-360**, a panoramic VQA dataset.

### CSV format

```
url,instruction,response
/path/to/image.jpg,What do you see?,"A wide panoramic scene..."
```

| Column | Description |
|--------|-------------|
| `url` | Local image path or Flickr URL |
| `instruction` | Question text |
| `response` | Ground-truth answer |

### Downloading the images (Flickr)

QuIC-360 images are hosted on Flickr and must be downloaded separately.

```bash
# 1) Obtain the source CSVs with Flickr URLs from the Refer360 release
#    (see the Refer360 repository below).

# 2) Download the images (16 parallel threads)
python scripts/download_quic360_images.py
#    Edit SAVE_DIR at the top of the script to change the target directory.
#    Default: data/quic360/images
```

> Refer360 dataset: <https://github.com/volkancirik/refer360>

### Pointing the CSVs at local images

If you downloaded the images to a custom path, rewrite the `url` column:

```python
import pandas as pd

NEW_IMG_DIR = "/your/path/to/quic360/images"

for split in ["train", "test"]:
    df = pd.read_csv(f"data/{split}.csv")
    # Flickr URL -> local path
    df["url"] = df["url"].apply(
        lambda u: f"{NEW_IMG_DIR}/{u.split('/')[-1]}" if u.startswith("http") else u
    )
    df.to_csv(f"data/{split}.csv", index=False)
```

Then place the CSVs where the configs expect them (see §5) or override the
paths in your config YAML:

```yaml
data_train_csv: "runs/baseline/_shared_data/train.csv"
data_test_csv:  "runs/baseline/_shared_data/test.csv"
```

### Statistics

| Split | Samples |
|-------|---------|
| Train | 7,929 |
| Test  | 5,349 |

---

## 4. Quick start

```bash
conda activate pano

# Smoke test (single GPU, ~5 min; requires the dataset from §3)
python scripts/smoke_panoadapt.py

# Train (LoRA, 1 epoch) — writes to runs/baseline/<experiment_name>/
python scripts/baseline_finetune.py \
    --config configs/baseline/panoadapt_vicreg_pairwise_internvl35_2b.yaml

# Evaluate (BLEU / METEOR / ROUGE-L / CIDEr / SPICE)
python scripts/baseline_eval.py \
    --config configs/baseline/panoadapt_vicreg_pairwise_internvl35_2b.yaml

# LLM-as-a-judge (multimodal GPT-based scoring; needs OPENAI_API_KEY)
python scripts/llm_judge_eval.py --input <predictions.csv>
```

### Selecting GPUs

```bash
export CUDA_VISIBLE_DEVICES=0     # GPU 0
export CUDA_VISIBLE_DEVICES=0,1   # multi-GPU
```

The full config-to-table mapping (which YAML reproduces which row) is in the
[README](../README.md#reproducing-table-1).

---

## 5. Directory layout

```
CORA-360/
├── src/cora/              # package (pip install -e .)
├── scripts/               # finetune / eval / inference entry points
├── configs/baseline/      # per-experiment YAML configs
└── runs/baseline/
    └── _shared_data/
        ├── train.csv      # QuIC-360 train (you provide this)
        └── test.csv       # QuIC-360 test  (you provide this)
```

Create the shared-data directory and drop your CSVs in:

```bash
mkdir -p runs/baseline/_shared_data
cp /your/path/to/quic360/train.csv runs/baseline/_shared_data/train.csv
cp /your/path/to/quic360/test.csv  runs/baseline/_shared_data/test.csv
```

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: cora` | package not installed | `pip install -e .` |
| `CUDA out of memory` | batch too large | `batch_size: 1` + `gradient_accumulation_steps: 4` |
| Gemma 3 `401 Unauthorized` | gated HF model | request access, then `huggingface-cli login` |
| SPICE evaluation fails | Java not installed | `sudo apt install default-jdk` |
| Image loading fails (`_MAX_RETRIES`) | wrong image path | check the `url` column in your CSV |
