# QuIC-360 Split Manifest

`quic360_split_manifest.csv` fixes the exact 7,929/5,349 train/test split used for the ECCV 2026 results.
It contains row order, image filename, and SHA-256 hashes of query and response text. It intentionally does
not redistribute images or annotations.

Run `scripts/verify_data_manifest.py` against locally obtained CSV files before training or evaluation.
`SOURCE_SHA256SUMS` records the hashes of the original path-bearing CSVs; these hashes are expected to differ
after replacing absolute image paths, while the path-independent manifest remains stable.
