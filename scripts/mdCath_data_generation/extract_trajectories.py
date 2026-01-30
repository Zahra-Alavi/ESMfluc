"""
Extract trajectory data from mdCATH HDF5 files.

Reads HDF5 files, extracts coordinates, topology, and sequence information,
and converts to formats compatible with pbxplore processing.

mdCATH structure:
- {domain_id}/{temperature}/{replica}/coords (n_frames, n_atoms, 3)
- {domain_id}/{temperature}/{replica}/dssp (n_frames, n_residues)
- {domain_id}/resid, resname, chain, element, z (topology)

Usage:
    python extract_trajectories.py --domain 12asA00
    python extract_trajectories.py --input-list downloaded_data/downloaded_files.txt
    python extract_trajectories.py --input-list file.txt --parallel 4
    python extract_trajectories.py --input-list file.txt --keep-intermediates
"""

import argparse
import json
import traceback
from pathlib import Path
from multiprocessing import Pool
import h5py
import numpy as np
from tqdm import tqdm
import portalocker

from constants import get_output_base, DIR_EXTRACTED_DATA, DIR_FAILED_DOMAINS, REPO_ID, REPO_TYPE

# Global variables that will be initialized in main()
EXTRACTS_DIR = None
FAILED_DOMAINS_DIR = None


def setup_directories():
    """Create necessary directories."""
    global EXTRACTS_DIR, FAILED_DOMAINS_DIR
    
    EXTRACTS_DIR.mkdir(parents=True, exist_ok=True)
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


def is_extracted(domain_id, extracts_dir):
    """Check if domain has been fully extracted."""
    domain_dir = extracts_dir / domain_id
    if not domain_dir.exists():
        return False
    
    # Check for metadata file
    metadata_file = domain_dir / "metadata.json"
    return metadata_file.exists()


def extract_trajectory_from_h5(h5_path, domain_id, output_dir):
    """
    Extract trajectory from mdCATH HDF5 file.
    
    mdCATH structure:
    - {domain_id}/{temperature}/{replica}/coords (n_frames, n_atoms, 3)
    - {domain_id}/{temperature}/{replica}/dssp (n_frames, n_residues)
    - {domain_id}/resid, resname, chain, element, z (topology)
    - {domain_id}/pdb (original PDB ID as string)
    
    Returns:
        dict: Metadata with extraction info (num_frames, num_residues, etc.)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "trajectories": [],
        "num_residues": 0,
        "temperatures": [],
        "num_replicates": 0,
    }
    
    try:
        with h5py.File(h5_path, 'r') as h5file:
            # Get domain group
            if domain_id not in h5file:
                raise ValueError(f"Domain {domain_id} not found in HDF5 file")
            
            domain_group = h5file[domain_id]
            
            # Extract topology information
            if 'resid' not in domain_group or 'resname' not in domain_group:
                raise ValueError("Missing topology data (resid/resname)")
            
            resids = domain_group['resid'][()]
            resnames = domain_group['resname'][()]
            chains = domain_group['chain'][()] if 'chain' in domain_group else None
            
            num_atoms = len(resids)
            num_residues = len(np.unique(resids))
            
            metadata["num_residues"] = int(num_residues)
            metadata["num_atoms"] = int(num_atoms)
            
            # Get PDB ID if available
            if 'pdb' in domain_group:
                pdb_data = domain_group['pdb'][()]
                pdb_id = pdb_data.decode() if isinstance(pdb_data, bytes) else str(pdb_data)
                metadata["pdb_id"] = pdb_id
            
            # Extract sequences and coordinates for each temperature/replica
            temperature_groups = [k for k in domain_group.keys() if k.isdigit()]
            temperatures = sorted([int(t) for t in temperature_groups])
            metadata["temperatures"] = temperatures
            
            all_coords = []
            all_dssp = []
            traj_count = 0
            
            for temp in temperatures:
                temp_group = domain_group[str(temp)]
                replicate_groups = sorted([k for k in temp_group.keys() if k.isdigit()])
                metadata["num_replicates"] = max(metadata["num_replicates"], len(replicate_groups))
                
                for rep_id in replicate_groups:
                    rep_group = temp_group[rep_id]
                    
                    # Get coordinates and DSSP
                    if 'coords' not in rep_group or 'dssp' not in rep_group:
                        raise ValueError(f"Missing coords/dssp in {domain_id}/{temp}/{rep_id}")
                    
                    coords = rep_group['coords'][()]  # (n_frames, n_atoms, 3)
                    dssp = rep_group['dssp'][()]      # (n_frames, n_residues)
                    
                    all_coords.append(coords)
                    all_dssp.append(dssp)
                    
                    metadata["trajectories"].append({
                        "temperature": int(temp),
                        "replicate": int(rep_id),
                        "num_frames": int(coords.shape[0]),
                    })
                    
                    traj_count += 1
            
            # Save combined data
            if all_coords:
                # Stack all trajectories
                combined_coords = np.concatenate(all_coords, axis=0)
                np.save(output_dir / "coordinates.npy", combined_coords)
                metadata["total_frames"] = int(combined_coords.shape[0])
                metadata["extracted"] = True
            else:
                raise ValueError("No trajectory data extracted")
            
        return metadata
        
    except Exception as e:
        metadata["extracted"] = False
        metadata["error"] = str(e)
        return metadata


def extract_domain(domain_info, resume=False, keep_intermediates=False, extracts_dir=None, failed_domains_dir=None):
    """Extract a single domain's trajectory data."""
    domain_id, h5_path = domain_info
    
    try:
        if resume and is_extracted(domain_id, extracts_dir):
            return {"status": "skipped", "domain_id": domain_id}
        
        output_dir = extracts_dir / domain_id
        
        # Extract trajectory
        metadata = extract_trajectory_from_h5(h5_path, domain_id, output_dir)
        
        if not metadata.get("extracted", False):
            error_msg = f"Failed to extract {domain_id}: {metadata.get('error', 'Unknown error')}"
            log_error(domain_id, error_msg, failed_domains_dir)
            return {"status": "failed", "domain_id": domain_id, "error": error_msg}
        
        # Save metadata
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {"status": "success", "domain_id": domain_id}
        
    except Exception as e:
        error_msg = f"Error extracting {domain_id}:\n{traceback.format_exc()}"
        log_error(domain_id, error_msg, failed_domains_dir)
        return {"status": "failed", "domain_id": domain_id, "error": str(e)}


def main():
    global EXTRACTS_DIR, FAILED_DOMAINS_DIR
    
    parser = argparse.ArgumentParser(
        description="Extract trajectory data from mdCATH HDF5 files"
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
        help="Extract single domain (e.g., 12asA00)"
    )
    group.add_argument(
        "--input-list",
        type=str,
        help="Path to file with downloaded domains list (downloaded_files.txt)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already extracted domains"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=min(4, __import__('multiprocessing').cpu_count()),
        help="Number of parallel processes (default: min(4, cpu_count))"
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep intermediate files (default: delete after processing)"
    )
    
    args = parser.parse_args()
    
    # Set up output directories
    output_base = get_output_base(args.output_base)
    EXTRACTS_DIR = output_base / DIR_EXTRACTED_DATA
    FAILED_DOMAINS_DIR = output_base / DIR_FAILED_DOMAINS
    
    setup_directories()
    
    # Collect domains to extract
    domains_to_extract = []
    
    if args.domain:
        # For single domain, need to find the h5 file path from downloaded_files.txt
        # or download it fresh using HfApi
        try:
            from huggingface_hub import hf_hub_download
            h5_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=f"data/mdcath_dataset_{args.domain}.h5",
                repo_type=REPO_TYPE
            )
            domains_to_extract = [(args.domain, h5_path)]
        except Exception as e:
            print(f"Error: Could not find or download {args.domain}: {e}")
            return
    
    else:  # --input-list
        if not Path(args.input_list).exists():
            print(f"Error: File {args.input_list} not found")
            return
        
        with open(args.input_list, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    domain_id, h5_path = parts[0], parts[1]
                    domains_to_extract.append((domain_id, h5_path))
    
    if not domains_to_extract:
        print("No domains to extract")
        return
    
    print(f"Extracting {len(domains_to_extract)} domains...")
    
    # Process with multiprocessing
    if args.parallel > 1:
        with Pool(processes=args.parallel) as pool:
            results = list(tqdm(
                pool.starmap(
                    extract_domain,
                    [(d, args.resume, args.keep_intermediates, EXTRACTS_DIR, FAILED_DOMAINS_DIR) for d in domains_to_extract]
                ),
                total=len(domains_to_extract),
                desc="Extracting trajectories"
            ))
    else:
        results = []
        for domain_info in tqdm(domains_to_extract, desc="Extracting trajectories"):
            result = extract_domain(domain_info, args.resume, args.keep_intermediates, EXTRACTS_DIR, FAILED_DOMAINS_DIR)
            results.append(result)
    
    # Summary
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    
    print(f"\n{'='*60}")
    print(f"Extraction Summary:")
    print(f"  Total domains: {len(domains_to_extract)}")
    print(f"  Successfully extracted: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Output base: {output_base}")
    print(f"  Extracted data directory: {EXTRACTS_DIR}")
    if failed_count > 0:
        print(f"  Failed domains log: {FAILED_DOMAINS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
