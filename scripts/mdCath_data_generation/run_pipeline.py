"""
Main pipeline orchestrator for mdCATH data processing.

Coordinates the download, extraction, Neq calculation, and export steps.

Usage:
    python run_pipeline.py --domain 12asA00 --skip-download          # Test single domain
    python run_pipeline.py --all --parallel 4                         # Full pipeline with parallelization
    python run_pipeline.py --resume --parallel 4 --skip-download      # Resume from interruption
    python run_pipeline.py --all --keep-intermediates                 # Keep extracted trajectories
"""

import argparse
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

from constants import get_output_base, DIR_DOWNLOADED_DATA


def run_command(cmd, description=""):
    """Run a shell command and handle errors."""
    if description:
        print(f"\n{'='*60}")
        print(f"Step: {description}")
        print(f"{'='*60}")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="mdCATH Neq Pipeline - Download, extract, calculate, and export"
    )
    
    # Domain selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--domain",
        type=str,
        help="Process single domain (e.g., 12asA00)"
    )
    group.add_argument(
        "--domains",
        nargs="+",
        help="Process specific domains"
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Process all mdCATH domains"
    )
    
    # Processing options
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (assume data already downloaded)"
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extraction step (assume data already extracted)"
    )
    parser.add_argument(
        "--skip-neq",
        action="store_true",
        help="Skip Neq calculation step"
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip export step"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from failures, skip completed steps"
    )
    
    # Parallelization
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1 for sequential)"
    )
    
    # Data handling
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep intermediate files (default: delete after processing)"
    )
    
    # Aggregation
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip aggregation step (default: aggregate after export)"
    )
    
    # Output base path
    parser.add_argument(
        "--output-base",
        type=str,
        default="data/mdcath",
        help="Base directory for all outputs (default: data/mdcath relative to workspace)"
    )
    
    args = parser.parse_args()
    
    # Calculate output base path
    output_base = get_output_base(args.output_base)
    
    # Build domain argument
    if args.domain:
        domain_args = ["--domain", args.domain]
    elif args.domains:
        domain_args = ["--domains"] + args.domains
    else:  # --all
        domain_args = ["--all"]
    
    # Step 1: Download
    if not args.skip_download:
        cmd = ["python", "data_download.py"] + domain_args + ["--output-base", str(output_base)]
        if args.resume:
            cmd.append("--resume")
        
        if not run_command(cmd, "Download mdCATH data from HuggingFace"):
            print("Download failed. Exiting.")
            sys.exit(1)
    else:
        print("Skipping download step")
    
    # Step 2: Extract trajectories
    if not args.skip_extract:
        if args.domain:
            # Need to find the downloaded file path
            extract_args = ["--domain", args.domain]
        elif args.domains:
            # Use input list approach
            domains_list_file = "domains_list.txt"
            with open(domains_list_file, 'w') as f:
                for domain in args.domains:
                    f.write(f"{domain}\n")
            extract_args = ["--input-list", domains_list_file]
        else:  # --all
            extract_args = ["--input-list", str(output_base / DIR_DOWNLOADED_DATA / "downloaded_files.txt")]
        
        cmd = ["python", "extract_trajectories.py"] + extract_args + ["--output-base", str(output_base)]
        if args.parallel > 1:
            cmd.extend(["--parallel", str(args.parallel)])
        if args.resume:
            cmd.append("--resume")
        if args.keep_intermediates:
            cmd.append("--keep-intermediates")
        
        if not run_command(cmd, "Extract trajectories from HDF5 files"):
            print("Extraction failed. Exiting.")
            sys.exit(1)
    else:
        print("Skipping extraction step")
    
    # Step 3: Calculate Neq
    if not args.skip_neq:
        if args.domain:
            neq_args = ["--domain", args.domain]
        elif args.domains:
            domains_list_file = "domains_list.txt"
            if Path(domains_list_file).exists():
                neq_args = ["--input-list", domains_list_file]
            else:
                # Create list from domains
                with open(domains_list_file, 'w') as f:
                    for domain in args.domains:
                        f.write(f"{domain}\n")
                neq_args = ["--input-list", domains_list_file]
        else:  # --all
            neq_args = ["--input-list", "extracted_data_domains.txt"]
            # Create list of extracted domains
            extracted_dir = output_base / "extracted_data"
            if extracted_dir.exists():
                with open("extracted_data_domains.txt", 'w') as f:
                    for domain_dir in extracted_dir.iterdir():
                        if domain_dir.is_dir():
                            f.write(f"{domain_dir.name}\n")
        
        cmd = ["python", "calculate_neq.py"] + neq_args + ["--output-base", str(output_base)]
        if args.parallel > 1:
            cmd.extend(["--parallel", str(args.parallel)])
        if args.resume:
            cmd.append("--resume")
        
        if not run_command(cmd, "Calculate Neq values"):
            print("Neq calculation failed. Exiting.")
            sys.exit(1)
    else:
        print("Skipping Neq calculation step")
    
    # Step 4: Export results
    if not args.skip_export:
        if args.domain:
            export_args = ["--domain", args.domain]
        elif args.domains:
            domains_list_file = "domains_list.txt"
            export_args = ["--input-list", domains_list_file]
        else:  # --all
            export_args = ["--all"]
        
        cmd = ["python", "export_results.py"] + export_args + ["--verify", "--output-base", str(output_base)]
        if args.keep_intermediates:
            cmd.append("--no-cleanup")
        
        if not run_command(cmd, "Export results to CSV files"):
            print("Export failed. Exiting.")
            sys.exit(1)
    else:
        print("Skipping export step")
    
    # Step 5: Aggregate sequences across domains
    if not args.skip_aggregate:
        # Only aggregate if we processed more than one domain
        if args.all or (args.domains and len(args.domains) > 1):
            cmd = ["python", "aggregate_sequences.py", "--output-base", str(output_base)]
            if args.domain:
                cmd.extend(["--domains", args.domain])
            elif args.domains:
                cmd.extend(["--domains"] + args.domains)
            
            if not run_command(cmd, "Aggregate sequences across domains"):
                print("Aggregation failed. Continuing anyway.")
        elif args.domain:
            # Single domain - still run aggregation for consistency
            cmd = ["python", "aggregate_sequences.py", "--output-base", str(output_base), "--domains", args.domain]
            if not run_command(cmd, "Aggregate sequences"):
                print("Aggregation failed. Continuing anyway.")
    else:
        print("Skipping aggregation step")
    
    # Final summary
    print(f"\n{'='*60}")
    print("Pipeline completed successfully!")
    print(f"{'='*60}")
    print("Output locations:")
    print(f"  Output base: {output_base}")
    print(f"  Downloaded data: {output_base / DIR_DOWNLOADED_DATA}")
    if args.keep_intermediates:
        print(f"  Extracted data (intermediate): {output_base / 'extracted_data'}")
        print(f"  Neq results (intermediate): {output_base / 'neq_results'}")
    else:
        print(f"  Intermediate files: Cleaned up ✓")
    print(f"  Final outputs: {output_base / 'outputs'}")
    print(f"  Aggregated sequences: {output_base / 'aggregated_sequences'}")
    print(f"  Failed domains: {output_base / 'failed_domains'}")
    print(f"{'='*60}")
    if not args.keep_intermediates:
        print("Note: Intermediate files (extracted_data, neq_results) were")
        print("automatically removed. Use --keep-intermediates to preserve them.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
