# Reproducing CORA

This document defines the ECCV 2026 artifact, not merely an example training run. The release is tied to
the Git tag `v2.0.0-eccv2026`, the split manifest in `data/manifests/`, and the three-seed reference metrics
in `results/table1_multiseed.json`.

## Reference environment

The camera-ready runs used Python 3.12.11, PyTorch 2.8.0+cu128, Transformers 4.56.2, PEFT 0.17.1,
CUDA 12.8, cuDNN 9.10, and two NVIDIA RTX 3090 GPUs with 24 GiB each. Each individual run uses one GPU.
Pinned direct runtime package versions are in `requirements-repro.txt`; transitive dependencies are resolved by pip.

```bash
conda env create -f environment.yml
conda activate cora-eccv2026
```

For a CUDA version other than 12.8, install the matching official PyTorch wheel first, then install the
remaining pinned requirements. Bitwise equality is not expected across GPU architectures. The release
acceptance check uses a metric tolerance of 0.002.

## Dataset

CORA uses the QuIC-360 split described by Maeda et al., Findings of EMNLP 2023. Images are not redistributed
in this repository. Obtain the dataset from its authors or an authorized source, create CSV files with
`url,instruction,response`, and verify the exact row order and content:

```bash
python scripts/verify_data_manifest.py \
  --train /path/to/train.csv \
  --test /path/to/test.csv
```

The released split contains 7,929 training and 5,349 test rows. Three test rows have an empty reference in
the source artifact; these are retained so sample counts and reported metrics remain identical. The manifest
stores image filenames plus SHA-256 hashes of queries and responses, avoiding redistribution of annotations.

## Evaluate a released adapter

See `MODEL_ZOO.md` for adapter IDs. The following command downloads the adapter from Hugging Face and runs
the matching config:

```bash
./reproduce.sh evaluate internvl3-2b-vicreg /path/to/test.csv
```

Evaluation writes `metrics.json` under `reproductions/<model-id>/eval/`. Verify it against the seed-42 reference:

```bash
./reproduce.sh verify internvl3-2b-vicreg reproductions/internvl3-2b-vicreg/eval/metrics.json
```

## Train from scratch

```bash
./reproduce.sh train internvl3-2b-vicreg /path/to/train.csv /path/to/test.csv
```

Training is one epoch with LoRA rank 32, gradient accumulation 4, and seed 42. The complete settings live in
the mapped YAML, and the script copies that YAML into the output directory. Repeat with seeds 1234 and 7 for
the three-seed table. Expected values are in `results/table1_multiseed.csv`.

## LLM judge

The lexical metrics are fully local. The optional LLM judge depends on a commercial, mutable service and is
therefore not used as the primary reproduction gate. The repository publishes the prompt implementation and
request parameters; archive raw judge outputs and the exact dated model identifier when rerunning it.

## Artifact integrity

Every Hugging Face adapter directory includes `SHA256SUMS`, its exact CORA YAML, seed-42 metrics, PEFT config,
and `adapter_model.safetensors`. The source tag should be cited together with the model repository revision.
