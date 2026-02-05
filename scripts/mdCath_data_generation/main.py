import os
import pandas as pd
import subprocess
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
DATA_ROOT = "../../data/mdcath"
DOMAIN_TXT = os.path.join(DATA_ROOT, "test_domain.txt")
H5_FOLDER = os.path.join(DATA_ROOT, "h5_files")
OUTPUT_ROOT = os.path.join(DATA_ROOT, "processed_data")
FINAL_CSV = os.path.join(DATA_ROOT, "mdcath_neq_dataset.csv")
MDCATH_SCRIPT = "convert_mdCath.py"
MAX_WORKERS = os.cpu_count() - 2 

HF_BASE_URL = "https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main/data"
RESIDUE_MAPPING = {'HSP': 'H', 'HSD': 'H', 'HSE': 'H', 'CYX': 'C', 'ASH': 'D', 'GLH': 'E'}

def download_h5(url, dest_path):
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

def get_1letter(resname):
    if resname in RESIDUE_MAPPING: return RESIDUE_MAPPING[resname]
    try: return convert_aa_code(resname)
    except: return 'X'

def get_aa_seq(pdb_path):
    try:
        u = mda.Universe(pdb_path)
        protein = u.select_atoms("protein")
        return "".join([get_1letter(r.resname) for r in protein.residues])
    except: return None

def calculate_temp_neq(xtc_files, pdb_file):
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

def cleanup_files(h5_path, domain_dir):
    """Removes the heavy raw data files to stay under disk quota."""
    if os.path.exists(h5_path):
        os.remove(h5_path)
    if os.path.exists(domain_dir):
        shutil.rmtree(domain_dir)

def process_single_domain(domain):
    domain_dir = os.path.join(OUTPUT_ROOT, domain)
    h5_filename = f"mdcath_dataset_{domain}.h5"
    h5_path = os.path.join(H5_FOLDER, h5_filename)
    
    try:
        os.makedirs(domain_dir, exist_ok=True)
        os.makedirs(H5_FOLDER, exist_ok=True)

        # 1. DOWNLOAD
        if not os.path.exists(h5_path):
            download_h5(f"{HF_BASE_URL}/{h5_filename}", h5_path)

        # 2. CONVERT
        convert_to_files(h5_path, basename=domain, output_dir=domain_dir)
        pdb_file = os.path.join(domain_dir, f"{domain}.pdb")
        if not os.path.exists(pdb_file): return []

        # 3. SEQUENCE & ANALYSIS
        aa_sequence = get_aa_seq(pdb_file)
        if not aa_sequence: return []
        
        xtc_files = glob.glob(os.path.join(domain_dir, "*.xtc"))
        neq_data = calculate_temp_neq(xtc_files, pdb_file)
        
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
        cleanup_files(h5_path, domain_dir)
            
        return domain_results
    except Exception as e:
        print(f"Error processing {domain}: {e}")
        cleanup_files(h5_path, domain_dir)
        return []

if __name__ == "__main__":
    print(f"Starting mdCATH data processing with {MAX_WORKERS} workers...")
    # --- Checkpointing Logic ---
    processed_domains = set()
    if os.path.exists(FINAL_CSV):
        existing_df = pd.read_csv(FINAL_CSV)
        processed_domains = set(existing_df['domain'].unique())
        print(f"Resuming: {len(processed_domains)} domains already finished.")

    with open(DOMAIN_TXT, "r") as f:
        domain_list = [d.strip() for d in f if d.strip() and d.strip() not in processed_domains]
    
    if not domain_list:
        print("All domains in list already processed!")
    else:
        print(f"Processing {len(domain_list)} new domains...")
        all_results = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(process_single_domain, domain_list))
            for sublist in results: all_results.extend(sublist)

        # Append new results to the CSV (Pivot and Save)
        if all_results:
            new_df = pd.DataFrame(all_results)
            pivot_df = new_df.pivot(index=['domain', 'sequence'], columns='temperature', values='neq_values').reset_index()
            pivot_df.columns = [f"neq_{c}" if str(c).isdigit() else c for c in pivot_df.columns]
            
            # If CSV exists, merge them. Otherwise, save new.
            if os.path.exists(FINAL_CSV):
                final_df = pd.concat([pd.read_csv(FINAL_CSV), pivot_df], ignore_index=True)
                final_df.to_csv(FINAL_CSV, index=False)
            else:
                pivot_df.to_csv(FINAL_CSV, index=False)
            print("Batch complete and saved.")