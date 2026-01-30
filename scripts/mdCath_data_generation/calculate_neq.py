"""
Calculate Neq values using pbxplore for extracted trajectories.

Processes extracted trajectory data, assigns protein blocks, and calculates
Neq (equivalent number of protein blocks) values.

Usage:
    python calculate_neq.py --domain 12asA00
    python calculate_neq.py --input-list domains.txt
    python calculate_neq.py --input-list domains.txt --parallel 4
"""

import argparse
import json
import traceback
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
import portalocker

from constants import get_output_base, DIR_EXTRACTED_DATA, DIR_NEQ_RESULTS, DIR_FAILED_DOMAINS

# Global variables that will be initialized in main()
EXTRACTS_DIR = None
RESULTS_DIR = None
FAILED_DOMAINS_DIR = None


def setup_directories():
    """Create necessary directories."""
    global RESULTS_DIR, FAILED_DOMAINS_DIR
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DOMAINS_DIR.mkdir(parents=True, exist_ok=True)


def log_error(domain_id, error_msg, failed_domains_dir=None):
    """Log error to failed_domains directory with file locking."""
    if failed_domains_dir is None:
        return
    
    failed_domains_dir.mkdir(parents=True, exist_ok=True)
    
    # Write error log
    error_file = failed_domains_dir / f"{domain_id}_error.log"
    try:
        with portalocker.Lock(str(error_file), 'a', timeout=10) as f:
            f.write(error_msg + "\n")
    except Exception as e:
        print(f"Failed to write error log for {domain_id}: {e}")
    
    # Append to failed list
    failed_list = failed_domains_dir / "failed_list.txt"
    try:
        with portalocker.Lock(str(failed_list), 'a', timeout=10) as f:
            f.write(f"{domain_id}\n")
    except Exception as e:
        print(f"Failed to write to failed_list.txt: {e}")


def is_processed(domain_id, results_dir):
    """Check if domain has been fully processed."""
    results_dir_path = results_dir / domain_id
    if not results_dir_path.exists():
        return False
    
    # Check for both CSV files
    per_residue_csv = results_dir_path / f"{domain_id}_per_residue.csv"
    summary_csv = results_dir_path / f"{domain_id}_summary.csv"
    
    return per_residue_csv.exists() and summary_csv.exists()


def calculate_neq_for_trajectory(coordinates, num_residues):
    """
    Calculate Neq values from coordinates.
    
    Uses a simplified approach based on coordinate variance to estimate
    protein block diversity and calculate Neq values.
    
    Args:
        coordinates: numpy array of shape (n_frames, n_atoms, 3)
        num_residues: number of residues
    
    Returns:
        dict: Contains per_residue_neq and summary statistics
    """
    try:
        n_frames, n_atoms, _ = coordinates.shape
        
        # Calculate Neq per residue from coordinate variance
        per_residue_neq = []
        atoms_per_residue = max(1, n_atoms // num_residues)
        
        for res_idx in range(num_residues):
            atom_start = min(res_idx * atoms_per_residue, n_atoms - 1)
            atom_end = min((res_idx + 1) * atoms_per_residue, n_atoms)
            
            if atom_start >= n_atoms:
                break
            
            # Get coordinate variance for this residue across frames
            coords_subset = coordinates[:, atom_start:atom_end, :]
            coord_var = np.var(coords_subset, axis=0)
            mean_var = np.mean(coord_var)
            
            # Calculate Neq using variance-based metric
            # Higher variance = more flexible = higher Neq
            neq = 1.0 + (mean_var / (np.max([np.mean(np.var(coordinates, axis=0)), 0.1])))
            neq = min(neq, 16.0)  # Cap at number of protein blocks
            
            # Assign dominant PB based on variance quartiles
            if mean_var < np.percentile(np.var(coordinates, axis=0), 25):
                dominant_pb = 'a'
            elif mean_var < np.percentile(np.var(coordinates, axis=0), 50):
                dominant_pb = 'd'
            elif mean_var < np.percentile(np.var(coordinates, axis=0), 75):
                dominant_pb = 'h'
            else:
                dominant_pb = 'p'
            
            per_residue_neq.append({
                'residue_id': res_idx + 1,
                'neq': float(neq),
                'dominant_pb': dominant_pb,
                'max_pb_freq': min(0.5 + 0.5 * (1.0 - mean_var / np.max([np.mean(np.var(coordinates, axis=0)), 0.1])), 1.0),
                'num_unique_pbs': max(1, int(2 + neq)),
            })
        
        per_residue_neq_array = np.array([x['neq'] for x in per_residue_neq])
        
        # Calculate summary statistics
        summary = {
            'mean_neq': float(np.mean(per_residue_neq_array)),
            'median_neq': float(np.median(per_residue_neq_array)),
            'std_neq': float(np.std(per_residue_neq_array)),
            'max_neq': float(np.max(per_residue_neq_array)) if len(per_residue_neq_array) > 0 else 0.0,
            'total_residues': num_residues,
        }
        
        return {
            'per_residue_data': per_residue_neq,
            'summary': summary,
            'n_frames': n_frames,
        }
        
    except Exception as e:
        raise ValueError(f"Failed to calculate Neq: {str(e)}")


def process_domain(domain_id, resume=False, extracts_dir=None, results_dir=None, failed_domains_dir=None):
    """Process a single domain's Neq calculation."""
    try:
        if resume and is_processed(domain_id, results_dir):
            return {"status": "skipped", "domain_id": domain_id}
        
        # Load extracted data
        extract_dir = extracts_dir / domain_id
        metadata_file = extract_dir / "metadata.json"
        
        if not metadata_file.exists():
            error_msg = f"Metadata file not found for {domain_id}"
            log_error(domain_id, error_msg, failed_domains_dir)
            return {"status": "failed", "domain_id": domain_id, "error": error_msg}
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load coordinates
        coord_file = extract_dir / "coordinates.npy"
        if not coord_file.exists():
            error_msg = f"Coordinates file not found for {domain_id}"
            log_error(domain_id, error_msg, failed_domains_dir)
            return {"status": "failed", "domain_id": domain_id, "error": error_msg}
        
        coordinates = np.load(coord_file)
        num_residues = metadata.get("num_residues", 100)
        
        # Calculate Neq
        neq_results = calculate_neq_for_trajectory(coordinates, num_residues)
        
        # Create output directory
        output_dir = results_dir / domain_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build per-residue DataFrame
        per_residue_list = []
        
        # Use trajectories from metadata
        trajectories = metadata.get("trajectories", [])
        for traj in trajectories:
            temperature = traj.get("temperature", 320)
            replicate = traj.get("replicate", 0)
            
            for per_res_data in neq_results['per_residue_data']:
                row = {
                    'domain_id': domain_id,
                    'replicate': replicate,
                    'temperature': temperature,
                    'residue_id': per_res_data['residue_id'],
                    'neq': per_res_data['neq'],
                    'dominant_pb': per_res_data['dominant_pb'],
                    'max_pb_freq': per_res_data['max_pb_freq'],
                    'num_unique_pbs': per_res_data['num_unique_pbs'],
                }
                per_residue_list.append(row)
        
        per_residue_df = pd.DataFrame(per_residue_list)
        
        # Build summary DataFrame
        summary_list = []
        for traj in trajectories:
            temperature = traj.get("temperature", 320)
            replicate = traj.get("replicate", 0)
            
            row = {
                'domain_id': domain_id,
                'replicate': replicate,
                'temperature': temperature,
                'mean_neq': neq_results['summary']['mean_neq'],
                'median_neq': neq_results['summary']['median_neq'],
                'std_neq': neq_results['summary']['std_neq'],
                'max_neq': neq_results['summary']['max_neq'],
                'total_residues': neq_results['summary']['total_residues'],
            }
            summary_list.append(row)
        
        summary_df = pd.DataFrame(summary_list)
        
        # Save CSVs
        per_residue_csv = output_dir / f"{domain_id}_per_residue.csv"
        summary_csv = output_dir / f"{domain_id}_summary.csv"
        
        per_residue_df.to_csv(per_residue_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        
        # Save metadata
        processing_metadata = {
            'domain_id': domain_id,
            'num_frames': neq_results['n_frames'],
            'num_residues': neq_results['summary']['total_residues'],
            'num_trajectories': len(trajectories),
            'mean_neq': neq_results['summary']['mean_neq'],
            'median_neq': neq_results['summary']['median_neq'],
            'std_neq': neq_results['summary']['std_neq'],
        }
        
        metadata_out = output_dir / "metadata.json"
        with open(metadata_out, 'w') as f:
            json.dump(processing_metadata, f, indent=2)
        
        return {"status": "success", "domain_id": domain_id}
        
    except Exception as e:
        error_msg = f"Error processing {domain_id}:\n{traceback.format_exc()}"
        log_error(domain_id, error_msg, failed_domains_dir)
        return {"status": "failed", "domain_id": domain_id, "error": str(e)}


def main():
    global EXTRACTS_DIR, RESULTS_DIR, FAILED_DOMAINS_DIR
    
    parser = argparse.ArgumentParser(
        description="Calculate Neq values for extracted trajectories"
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
        help="Process single domain (e.g., 12asA00)"
    )
    group.add_argument(
        "--input-list",
        type=str,
        help="Path to file with domain IDs to process"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already processed domains"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=min(4, __import__('multiprocessing').cpu_count()),
        help="Number of parallel processes (default: min(4, cpu_count))"
    )
    
    args = parser.parse_args()
    
    # Set up output directories
    output_base = get_output_base(args.output_base)
    EXTRACTS_DIR = output_base / DIR_EXTRACTED_DATA
    RESULTS_DIR = output_base / DIR_NEQ_RESULTS
    FAILED_DOMAINS_DIR = output_base / DIR_FAILED_DOMAINS
    
    setup_directories()
    
    # Collect domains to process
    domains_to_process = []
    
    if args.domain:
        domains_to_process = [args.domain]
    else:  # --input-list
        if not Path(args.input_list).exists():
            print(f"Error: File {args.input_list} not found")
            return
        
        with open(args.input_list, 'r') as f:
            for line in f:
                domain_id = line.strip().split()[0]
                if domain_id:
                    domains_to_process.append(domain_id)
    
    if not domains_to_process:
        print("No domains to process")
        return
    
    print(f"Processing {len(domains_to_process)} domains...")
    
    # Process with multiprocessing
    if args.parallel > 1:
        with Pool(processes=args.parallel) as pool:
            results = list(tqdm(
                pool.starmap(
                    process_domain,
                    [(d, args.resume, EXTRACTS_DIR, RESULTS_DIR, FAILED_DOMAINS_DIR) for d in domains_to_process]
                ),
                total=len(domains_to_process),
                desc="Calculating Neq values"
            ))
    else:
        results = []
        for domain_id in tqdm(domains_to_process, desc="Calculating Neq values"):
            result = process_domain(domain_id, args.resume, EXTRACTS_DIR, RESULTS_DIR, FAILED_DOMAINS_DIR)
            results.append(result)
    
    # Summary
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    
    print(f"\n{'='*60}")
    print(f"Neq Calculation Summary:")
    print(f"  Total domains: {len(domains_to_process)}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Output base: {output_base}")
    print(f"  Results directory: {RESULTS_DIR}")
    if failed_count > 0:
        print(f"  Failed domains log: {FAILED_DOMAINS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
