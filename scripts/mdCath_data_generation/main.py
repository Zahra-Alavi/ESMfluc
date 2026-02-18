import os
import argparse
import pandas as pd
import glob
import shutil
import numpy as np
import requests
import pbxplore as pbx
import MDAnalysis as mda
from MDAnalysis.lib.util import convert_aa_code
from concurrent.futures import ProcessPoolExecutor
from convert_mdCath import convert_to_files

# --- Configuration ---
data_root = "../../data/mdcath"
domain_list_file = "test_domain.txt"
h5_folder = "h5_files"
output_root = "processed_data"
final_csv = "mdcath_neq_dataset.csv" 
MAX_WORKERS = os.cpu_count() - 2 

HF_BASE_URL = "https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main/data"
RESIDUE_MAPPING = {'HSP': 'H', 'HSD': 'H', 'HSE': 'H', 'CYX': 'C', 'ASH': 'D', 'GLH': 'E'}

def arg_parser():
    parser = argparse.ArgumentParser(
        description="Compute Neq values from mdCATH simulation data using Pbxplore and save to CSV."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        help=f"Root directory for mdCATH data, containing domain list and where outputs will be saved. Defaults to {data_root}.",
        default=data_root,
    )
    parser.add_argument(
        "--domain_list_file",
        type=str,
        help=f"Name of the text file containing the list of domain IDs to process, located in data_root. Defaults to {domain_list_file}.",
        default=domain_list_file,
    )
    parser.add_argument(
        "--h5_folder",
        type=str,
        help=f"Subdirectory within data_root where H5 files will be stored. Defaults to {h5_folder}.",
        default=h5_folder,
    )
    parser.add_argument(
        "--output_root",
        type=str,
        help=f"Subdirectory within data_root where processed outputs will be saved. Defaults to {output_root}.",
        default=output_root,
    )
    parser.add_argument(
        "--final_csv",
        type=str,
        help=f"Name of the final CSV file to save results to, located in data_root. Defaults to {final_csv}.",
        default=final_csv,
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        help=f"Maximum number of parallel workers for processing domains. Defaults to number of CPU cores minus 2: {MAX_WORKERS}.",
        default=MAX_WORKERS,
    )
    
    return parser.parse_args()

def _download_h5(url, dest_path):
    try:
        print(f"Downloading {url} to {dest_path}...")
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except requests.exceptions.HTTPError as e:
        print(f"Failed to download: {url} - Error: {e}")
        raise 

def _get_1letter(resname):
    if resname in RESIDUE_MAPPING: return RESIDUE_MAPPING[resname]
    try: return convert_aa_code(resname)
    except: return 'X'

def _get_aa_seq(pdb_path):
    try:
        u = mda.Universe(pdb_path)
        protein = u.select_atoms("protein")
        return "".join([_get_1letter(r.resname) for r in protein.residues])
    except: return None

def _calculate_temp_neq(xtc_files, pdb_file):
    """Groups XTCs by temperature and calculates the mean Neq."""
    temp_groups = {}
    for f in xtc_files:
        temp = f.split('_')[-2]
        temp_groups.setdefault(temp, []).append(f)
    
    results = []
    for temp, files in temp_groups.items():
        all_replica_neqs = []
        for xtc in files:
            # Generate PB sequences for each frame in the trajectory
            pb_seqs = [pbx.assign(chain.get_phi_psi_angles()) 
                       for _, chain in pbx.chains_from_trajectory(xtc, pdb_file)]
            if pb_seqs:
                # Get Neq for this specific replica
                counts = pbx.analysis.count_matrix(pb_seqs)
                all_replica_neqs.append(pbx.analysis.compute_neq(counts))
        
        if all_replica_neqs:
            results.append({
                "temp": temp,
                "avg_neq": np.mean(all_replica_neqs, axis=0).tolist()
            })
    return results

def _cleanup_files(h5_path, domain_dir):
    """Removes the heavy raw data files to stay under disk quota."""
    if os.path.exists(h5_path):
        os.remove(h5_path)
    if os.path.exists(domain_dir):
        shutil.rmtree(domain_dir)

def process_single_domain(domain):
    domain_dir = os.path.join(output_root, domain)
    h5_filename = f"mdcath_dataset_{domain}.h5"
    h5_path = os.path.join(h5_folder, h5_filename)
    
    try:
        os.makedirs(domain_dir, exist_ok=True)
        os.makedirs(h5_folder, exist_ok=True)

        # 1. DOWNLOAD
        if not os.path.exists(h5_path):
            _download_h5(f"{HF_BASE_URL}/{h5_filename}", h5_path)

        # 2. CONVERT
        convert_to_files(h5_path, basename=domain, output_dir=domain_dir)
        pdb_file = os.path.join(domain_dir, f"{domain}.pdb")
        if not os.path.exists(pdb_file): return []

        # 3. SEQUENCE & ANALYSIS
        aa_sequence = _get_aa_seq(pdb_file)
        if not aa_sequence: return []
        
        xtc_files = glob.glob(os.path.join(domain_dir, "*.xtc"))
        neq_data = _calculate_temp_neq(xtc_files, pdb_file)
        
        # 4. DATA WRAPPING
        domain_results = []
        for item in neq_data:
            domain_results.append({
                "domain": domain,
                "sequence": aa_sequence,
                "temperature": item["temp"],
                "neq_values": item["avg_neq"]
            })

        # 5. CLEANUP
        _cleanup_files(h5_path, domain_dir)
            
        return domain_results
    except Exception as e:
        print(f"Error processing {domain}: {e}")
        _cleanup_files(h5_path, domain_dir)
        return []

if __name__ == "__main__":
    print(f"Starting mdCATH data processing with {MAX_WORKERS} workers...")
    
    # --- Argument Parsing ---
    args = arg_parser()
    data_root = args.data_root
    domain_list_file = os.path.join(data_root, args.domain_list_file)
    h5_folder = os.path.join(data_root, args.h5_folder)
    output_root = os.path.join(data_root, args.output_root)
    final_csv = os.path.join(data_root, args.final_csv)
    max_workers = args.max_workers

    # --- Checkpointing Logic ---
    processed_domains = set()
    if os.path.exists(final_csv):
        existing_df = pd.read_csv(final_csv)
        processed_domains = set(existing_df['domain'].unique())
        print(f"Resuming: {len(processed_domains)} domains already finished.")

    with open(domain_list_file, "r") as f:
        domain_list = [d.strip() for d in f if d.strip() and d.strip() not in processed_domains]
    
    if not domain_list:
        print("All domains in list already processed!")
    else:
        CHUNK_SIZE = 50
        print(f"Processing {len(domain_list)} domains in chunks of {CHUNK_SIZE}...")

        for i in range(0, len(domain_list), CHUNK_SIZE):
            current_chunk = domain_list[i : i + CHUNK_SIZE]
            print(f"--- Starting Chunk {i//CHUNK_SIZE + 1}: {current_chunk[0]} to {current_chunk[-1]} ---")
            
            chunk_results = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_single_domain, current_chunk))
                for sublist in results:
                    chunk_results.extend(sublist)

            if chunk_results:
                new_batch_df = pd.DataFrame(chunk_results)
                
                pivot_df = new_batch_df.pivot(
                    index=['domain', 'sequence'], 
                    columns='temperature', 
                    values='neq_values'
                ).reset_index()
                
                pivot_df.columns = [f"neq_{c}" if str(c).isdigit() else c for c in pivot_df.columns]
                
                if os.path.exists(final_csv):
                    existing_df = pd.read_csv(final_csv)
                    final_df = pd.concat([existing_df, pivot_df], ignore_index=True, sort=False)
                    final_df.drop_duplicates(subset=['domain'], inplace=True)
                    final_df.to_csv(final_csv, index=False)
                else:
                    pivot_df.to_csv(final_csv, index=False)
                
                print(f"Chunk {i//CHUNK_SIZE + 1} saved successfully.")
        print("All chunks processed. Final dataset saved to:", final_csv)