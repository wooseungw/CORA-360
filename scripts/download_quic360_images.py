"""Download QuIC-360 Flickr images to local storage."""
import os
import hashlib
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

SAVE_DIR = "data/quic360/images"
CSV_DIR = "data/quic360"
SPLITS = ["train", "valid", "test"]
MAX_WORKERS = 16
TIMEOUT = 30


def url_to_filename(url: str) -> str:
    """Convert Flickr URL to a stable local filename."""
    # Extract the Flickr image ID from URL for readable filenames
    # e.g. https://live.staticflickr.com/8380/29237818512_ebfa8f5d04_f.jpg -> 29237818512_ebfa8f5d04_f.jpg
    basename = url.split("/")[-1]
    if basename:
        return basename
    # Fallback: hash the URL
    return hashlib.md5(url.encode()).hexdigest() + ".jpg"


def download_one(url: str, save_dir: str) -> tuple[str, bool, str]:
    """Download a single image. Returns (url, success, local_path_or_error)."""
    fname = url_to_filename(url)
    local_path = os.path.join(save_dir, fname)
    
    # Skip if already downloaded
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return (url, True, local_path)
    
    try:
        r = requests.get(url, timeout=TIMEOUT, stream=True)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return (url, True, local_path)
    except Exception as e:
        return (url, False, str(e))


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Collect all unique Flickr URLs
    all_urls = set()
    for split in SPLITS:
        df = pd.read_csv(os.path.join(CSV_DIR, f"{split}.csv"))
        flickr = df[df["url"].str.startswith("http")]["url"].unique()
        all_urls.update(flickr)
    
    print(f"Total unique Flickr URLs: {len(all_urls)}")
    
    # Download with thread pool
    success = 0
    failed = 0
    url_to_local = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_one, url, SAVE_DIR): url for url in all_urls}
        
        with tqdm(total=len(futures), desc="Downloading", unit="img") as pbar:
            for future in as_completed(futures):
                url, ok, result = future.result()
                if ok:
                    success += 1
                    url_to_local[url] = result
                else:
                    failed += 1
                    if failed <= 10:
                        tqdm.write(f"  FAILED: {url[:60]}... -> {result}")
                pbar.update(1)
                pbar.set_postfix(ok=success, fail=failed)
    
    print(f"\nDownload complete: {success} success, {failed} failed")
    
    # Now rewrite CSVs with local paths
    for split in SPLITS:
        csv_path = os.path.join(CSV_DIR, f"{split}.csv")
        df = pd.read_csv(csv_path)
        
        # Map Flickr URLs to local paths
        def map_url(url):
            if url in url_to_local:
                return url_to_local[url]
            elif url.startswith("http"):
                # Try to construct local path even if not in map
                fname = url_to_filename(url)
                local = os.path.join(SAVE_DIR, fname)
                if os.path.exists(local):
                    return local
            return url  # Keep original if not downloaded
        
        df["url"] = df["url"].apply(map_url)
        
        # Save updated CSV
        out_path = os.path.join(CSV_DIR, f"{split}.csv")
        df.to_csv(out_path, index=False)
        
        # Stats
        local_count = df["url"].str.startswith("/").sum()
        remote_count = df["url"].str.startswith("http").sum()
        print(f"  {split}: {local_count} local, {remote_count} still remote")
    
    print("Done!")


if __name__ == "__main__":
    main()
