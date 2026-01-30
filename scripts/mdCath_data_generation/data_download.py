"""
Download mdCATH dataset files from HuggingFace Hub.

Usage:
    python data_download.py --domain 12asA00              # Download single domain
    python data_download.py --domains 12asA00 1mba00      # Download specific domains
    python data_download.py --all                          # Download all domains
    python data_download.py --all --resume                 # Download all, skip existing
"""

import argparse
import re
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download, HfApi

from constants import REPO_ID, REPO_TYPE, get_output_base, DIR_DOWNLOADED_DATA

# Global variables that will be initialized in main()
DOWNLOADS_DIR = None
DOWNLOADS_LIST = None


def get_available_domains():
    """Get list of all available mdCATH domain IDs from HuggingFace."""
    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset")
    
    # Files are in data/ subdirectory
    domain_pattern = r'data/mdcath_dataset_(\w+)\.h5'
    domains = []
    
    for filename in all_files:
        match = re.search(domain_pattern, filename)
        if match:
            domains.append(match.group(1))
    
    return sorted(domains)


def is_downloaded(domain_id):
    """Check if domain has been downloaded."""
    if not DOWNLOADS_LIST.exists():
        return False
    
    with open(DOWNLOADS_LIST, 'r') as f:
        downloaded = f.read().splitlines()
    
    return domain_id in downloaded


def record_download(domain_id, file_path):
    """Record downloaded domain ID and file path."""
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(DOWNLOADS_LIST, 'a') as f:
        f.write(f"{domain_id}\t{file_path}\n")


def download_domain(domain_id, resume=False):
    """Download a single mdCATH domain."""
    if resume and is_downloaded(domain_id):
        print(f"Skipping {domain_id} (already downloaded)")
        return True
    
    # Files are in data/ subdirectory
    filename = f"data/mdcath_dataset_{domain_id}.h5"
    
    try:
        local_file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset"
        )
        record_download(domain_id, local_file_path)
        print(f"Downloaded {domain_id}: {local_file_path}")
        return True
    except Exception as e:
        print(f"Error downloading {domain_id}: {e}")
        return False


def main():
    global DOWNLOADS_DIR, DOWNLOADS_LIST
    
    parser = argparse.ArgumentParser(
        description="Download mdCATH dataset files from HuggingFace Hub"
    )
    
    parser.add_argument(
        "--output-base",
        type=str,
        default="data/mdcath",
        help="Base directory for all outputs (default: data/mdcath relative to workspace)"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--domain",
        type=str,
        help="Download single domain (e.g., 12asA00)"
    )
    group.add_argument(
        "--domains",
        nargs="+",
        help="Download specific domains (e.g., 12asA00 1mba00)"
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Download all mdCATH domains"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already downloaded domains"
    )
    
    args = parser.parse_args()
    
    # Set up output directories
    output_base = get_output_base(args.output_base)
    DOWNLOADS_DIR = output_base / DIR_DOWNLOADED_DATA
    DOWNLOADS_LIST = DOWNLOADS_DIR / "downloaded_files.txt"
    
    # Determine which domains to download
    if args.domain:
        domains = [args.domain]
    elif args.domains:
        domains = args.domains
    else:  # --all
        print("Fetching list of available domains...")
        domains = get_available_domains()
        print(f"Found {len(domains)} domains to download")
    
    # Download domains with progress bar
    downloaded_count = 0
    failed_domains = []
    
    with tqdm(total=len(domains), desc="Downloading domains") as pbar:
        for domain_id in domains:
            success = download_domain(domain_id, resume=args.resume)
            if success:
                downloaded_count += 1
            else:
                failed_domains.append(domain_id)
            pbar.update(1)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Download Summary:")
    print(f"  Total domains: {len(domains)}")
    print(f"  Successfully downloaded: {downloaded_count}")
    print(f"  Failed: {len(failed_domains)}")
    
    if failed_domains:
        print(f"  Failed domains: {', '.join(failed_domains)}")
    
    print(f"  Output base: {output_base}")
    print(f"  Downloaded files list: {DOWNLOADS_LIST}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()