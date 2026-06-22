#!/usr/bin/env python3
"""Re-download corrupted JPEG images from Flickr.

Scans the quic360 image directory for truncated JPEGs (missing FFD9 end marker),
looks up their original Flickr URLs from the raw CSV, and re-downloads them.
"""

import os
import sys
import glob
import time
import requests
import pandas as pd
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────
IMG_DIR = "data/quic360/images"
RAW_CSV_DIR = Path(__file__).resolve().parent.parent / "legacy/CORA/data/raw/QuIC360"
TIMEOUT = 60
MAX_RETRIES = 3


def find_corrupted_jpegs(img_dir: str) -> list[str]:
    """Find JPEGs missing the FFD9 end-of-image marker."""
    bad = []
    for f in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
        with open(f, "rb") as fp:
            fp.seek(-2, 2)
            if fp.read(2) != b"\xff\xd9":
                bad.append(os.path.basename(f))
    return bad


def build_url_map(raw_csv_dir: Path) -> dict[str, str]:
    """Build filename → Flickr URL mapping from raw CSVs."""
    url_map: dict[str, str] = {}
    for split in ["train", "valid", "test", "downtest"]:
        p = raw_csv_dir / f"{split}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "url" not in df.columns:
            continue
        for url in df["url"].dropna().unique():
            if url.startswith("http"):
                fname = url.split("/")[-1]
                url_map[fname] = url
    return url_map


def download_with_retry(url: str, dest: str, retries: int = MAX_RETRIES) -> bool:
    """Download URL to dest, overwriting existing file."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT, stream=True)
            r.raise_for_status()
            tmp = dest + ".tmp"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
            # Verify JPEG end marker
            with open(tmp, "rb") as f:
                f.seek(-2, 2)
                if f.read(2) != b"\xff\xd9":
                    print(f"  ⚠ Downloaded but still truncated: {os.path.basename(dest)}")
                    os.remove(tmp)
                    return False
            os.replace(tmp, dest)
            return True
        except Exception as e:
            print(f"  Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    return False


def main():
    print("Scanning for corrupted JPEGs...")
    bad_files = find_corrupted_jpegs(IMG_DIR)
    print(f"Found {len(bad_files)} corrupted images\n")

    if not bad_files:
        print("All images are intact!")
        return

    print("Building URL map from raw CSVs...")
    url_map = build_url_map(RAW_CSV_DIR)
    print(f"URL map has {len(url_map)} entries\n")

    success = 0
    failed = 0
    no_url = 0

    for fname in bad_files:
        url = url_map.get(fname)
        if not url:
            print(f"✗ No URL found for {fname}")
            no_url += 1
            continue

        dest = os.path.join(IMG_DIR, fname)
        old_size = os.path.getsize(dest)
        print(f"↓ {fname} ({old_size:,} bytes) ← {url[:80]}...")

        # Remove corrupted file first
        os.remove(dest)

        if download_with_retry(url, dest):
            new_size = os.path.getsize(dest)
            print(f"  ✓ OK ({new_size:,} bytes)")
            success += 1
        else:
            print(f"  ✗ FAILED")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {success} fixed, {failed} failed, {no_url} no URL")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
