"""
Export Neq results to CSV files.

Validates processed results, exports to per-domain CSV files with metadata.

Usage:
    python export_results.py --domain 12asA00
    python export_results.py --input-list domains.txt
    python export_results.py --input-list domains.txt --verify
"""

import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from constants import (
    get_output_base, DIR_NEQ_RESULTS, DIR_OUTPUTS, DIR_EXTRACTED_DATA,
    REQUIRED_PER_RESIDUE_COLUMNS, REQUIRED_SUMMARY_COLUMNS,
    REQUIRED_METADATA_KEYS, MIN_ROWS_PER_RESIDUE, MIN_ROWS_SUMMARY
)

# Global variables that will be initialized in main()
RESULTS_DIR = None
OUTPUTS_DIR = None
EXTRACTED_DIR = None


def setup_directories():
    """Create necessary directories."""
    global OUTPUTS_DIR
    
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def validate_csv_integrity(csv_path, expected_columns, min_rows=1):
    """Validate CSV file integrity."""
    try:
        if not csv_path.exists():
            return False, "File does not exist"
        
        df = pd.read_csv(csv_path)
        
        # Check if empty
        if len(df) < min_rows:
            return False, f"File has fewer than {min_rows} rows"
        
        # Check columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"
        
        return True, "Valid"
    except Exception as e:
        return False, str(e)


def validate_domain_results(domain_id):
    """Validate that domain results are complete and correct."""
    results_dir = RESULTS_DIR / domain_id
    
    if not results_dir.exists():
        return False, "Results directory does not exist"
    
    # Check per-residue CSV
    per_residue_csv = results_dir / f"{domain_id}_per_residue.csv"
    per_res_valid, per_res_msg = validate_csv_integrity(
        per_residue_csv,
        expected_columns=REQUIRED_PER_RESIDUE_COLUMNS,
        min_rows=MIN_ROWS_PER_RESIDUE
    )
    
    if not per_res_valid:
        return False, f"Per-residue CSV invalid: {per_res_msg}"
    
    # Check summary CSV
    summary_csv = results_dir / f"{domain_id}_summary.csv"
    summary_valid, summary_msg = validate_csv_integrity(
        summary_csv,
        expected_columns=REQUIRED_SUMMARY_COLUMNS,
        min_rows=MIN_ROWS_SUMMARY
    )
    
    if not summary_valid:
        return False, f"Summary CSV invalid: {summary_msg}"
    
    # Check metadata
    metadata_file = results_dir / "metadata.json"
    if not metadata_file.exists():
        return False, "Metadata file does not exist"
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        missing_keys = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
        if missing_keys:
            return False, f"Metadata missing keys: {missing_keys}"
    except Exception as e:
        return False, f"Metadata invalid: {str(e)}"
    
    return True, "Valid"


def export_domain_results(domain_id, verify=False, cleanup=True):
    """Export results for a single domain."""
    try:
        # Validate if requested
        if verify:
            is_valid, msg = validate_domain_results(domain_id)
            if not is_valid:
                return {"status": "invalid", "domain_id": domain_id, "error": msg}
        
        results_dir = RESULTS_DIR / domain_id
        output_dir = OUTPUTS_DIR / domain_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy per-residue CSV
        per_residue_src = results_dir / f"{domain_id}_per_residue.csv"
        per_residue_dst = output_dir / f"{domain_id}_per_residue.csv"
        if per_residue_src.exists():
            df = pd.read_csv(per_residue_src)
            df.to_csv(per_residue_dst, index=False)
        else:
            return {"status": "failed", "domain_id": domain_id, "error": "Per-residue CSV not found"}
        
        # Copy summary CSV
        summary_src = results_dir / f"{domain_id}_summary.csv"
        summary_dst = output_dir / f"{domain_id}_summary.csv"
        if summary_src.exists():
            df = pd.read_csv(summary_src)
            df.to_csv(summary_dst, index=False)
        else:
            return {"status": "failed", "domain_id": domain_id, "error": "Summary CSV not found"}
        
        # Copy metadata
        metadata_src = results_dir / "metadata.json"
        metadata_dst = output_dir / "metadata.json"
        if metadata_src.exists():
            with open(metadata_src, 'r') as f:
                metadata = json.load(f)
            with open(metadata_dst, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Create summary report
        with open(metadata_src, 'r') as f:
            metadata = json.load(f)
        
        report_file = output_dir / "README.md"
        report_content = f"""# mdCATH Neq Results for {domain_id}

## Overview
- **Domain ID:** {domain_id}
- **Number of Frames:** {metadata.get('num_frames', 'N/A')}
- **Number of Residues:** {metadata.get('num_residues', 'N/A')}
- **Number of Trajectories:** {metadata.get('num_trajectories', 'N/A')}

## Statistics
- **Mean Neq:** {metadata.get('mean_neq', 'N/A'):.4f}
- **Median Neq:** {metadata.get('median_neq', 'N/A'):.4f}
- **Std Neq:** {metadata.get('std_neq', 'N/A'):.4f}

## Files
- `{domain_id}_per_residue.csv`: Per-residue Neq values for each trajectory
- `{domain_id}_summary.csv`: Summary statistics for each trajectory
- `metadata.json`: Processing metadata

## Column Descriptions

### Per-Residue CSV
- `domain_id`: CATH domain identifier
- `replicate`: Replica number (1-5)
- `temperature`: Temperature in Kelvin (320, 350, 380, 410, 450)
- `residue_id`: Residue position in the protein
- `neq`: Equivalent number of protein blocks (higher = more flexible)
- `dominant_pb`: Most frequent protein block at this position
- `max_pb_freq`: Maximum frequency of any protein block
- `num_unique_pbs`: Number of unique protein blocks observed

### Summary CSV
- `domain_id`: CATH domain identifier
- `replicate`: Replica number (1-5)
- `temperature`: Temperature in Kelvin
- `mean_neq`: Average Neq across all residues
- `median_neq`: Median Neq across all residues
- `std_neq`: Standard deviation of Neq
- `max_neq`: Maximum Neq value
- `total_residues`: Total number of residues analyzed
"""
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        # Clean up intermediate files if requested
        if cleanup:
            # Remove extracted_data/{domain_id}
            extracted_domain_dir = EXTRACTED_DIR / domain_id
            if extracted_domain_dir.exists():
                shutil.rmtree(extracted_domain_dir)
            
            # Remove neq_results/{domain_id}
            if results_dir.exists():
                shutil.rmtree(results_dir)
        
        return {"status": "success", "domain_id": domain_id}
        
    except Exception as e:
        return {"status": "failed", "domain_id": domain_id, "error": str(e)}


def main():
    global RESULTS_DIR, OUTPUTS_DIR, EXTRACTED_DIR
    
    parser = argparse.ArgumentParser(
        description="Export Neq results to CSV files"
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
        help="Export results for single domain (e.g., 12asA00)"
    )
    group.add_argument(
        "--input-list",
        type=str,
        help="Path to file with domain IDs to export"
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Export all results in {RESULTS_DIR}"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify results before exporting"
    )
    
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep intermediate files (extracted_data and neq_results)"
    )
    
    args = parser.parse_args()
    
    # Set up output directories
    output_base = get_output_base(args.output_base)
    RESULTS_DIR = output_base / DIR_NEQ_RESULTS
    OUTPUTS_DIR = output_base / DIR_OUTPUTS
    EXTRACTED_DIR = output_base / DIR_EXTRACTED_DATA
    
    setup_directories()
    
    # Collect domains to export
    domains_to_export = []
    
    if args.domain:
        domains_to_export = [args.domain]
    elif args.input_list:
        if not Path(args.input_list).exists():
            print(f"Error: File {args.input_list} not found")
            return
        
        with open(args.input_list, 'r') as f:
            for line in f:
                domain_id = line.strip().split()[0]
                if domain_id:
                    domains_to_export.append(domain_id)
    else:  # --all
        if not RESULTS_DIR.exists():
            print(f"Error: Results directory {RESULTS_DIR} not found")
            return
        
        domains_to_export = [d.name for d in RESULTS_DIR.iterdir() if d.is_dir()]
    
    if not domains_to_export:
        print("No domains to export")
        return
    
    print(f"Exporting {len(domains_to_export)} domains...")
    
    # Export with progress bar
    cleanup = not args.no_cleanup
    results = []
    for domain_id in tqdm(domains_to_export, desc="Exporting results"):
        result = export_domain_results(domain_id, verify=args.verify, cleanup=cleanup)
        results.append(result)
    results = []
    for domain_id in tqdm(domains_to_export, desc="Exporting results"):
        result = export_domain_results(domain_id, verify=args.verify)
        results.append(result)
    
    # Summary
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    invalid_count = sum(1 for r in results if r["status"] == "invalid")
    
    print(f"\n{'='*60}")
    print(f"Export Summary:")
    print(f"  Total domains: {len(domains_to_export)}")
    print(f"  Successfully exported: {success_count}")
    print(f"  Failed: {failed_count}")
    if args.verify:
        print(f"  Invalid: {invalid_count}")
    print(f"  Output base: {output_base}")
    print(f"  Output directory: {OUTPUTS_DIR}")
    print(f"{'='*60}")
    
    # Show any errors
    if failed_count > 0 or invalid_count > 0:
        print("\nErrors:")
        for result in results:
            if result["status"] in ["failed", "invalid"]:
                print(f"  {result['domain_id']}: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
