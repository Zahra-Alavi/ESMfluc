"""
Centralized constants for mdCATH data generation pipeline.
"""

from pathlib import Path

# ============================================================================
# Output Base Directory Configuration
# ============================================================================

def get_default_output_base():
    """
    Get default output base path: ../data/mdcath relative to script location.
    
    Returns:
        Path: Absolute path to data/mdcath directory
    """
    return Path(__file__).parent.parent.parent / "data" / "mdcath"


def get_output_base(output_base_arg: str = None) -> Path:
    """
    Resolve output base path from argument or use default.
    
    Args:
        output_base_arg: Command-line argument value (or None for default)
        
    Returns:
        Path: Absolute path to output base directory
    """
    if output_base_arg is None or output_base_arg == "data/mdcath":
        return get_default_output_base()
    else:
        return Path(output_base_arg).resolve()


# ============================================================================
# Directory Names (subdirectories within output_base)
# ============================================================================

DIR_DOWNLOADED_DATA = "downloaded_data"
DIR_EXTRACTED_DATA = "extracted_data"
DIR_NEQ_RESULTS = "neq_results"
DIR_OUTPUTS = "outputs"
DIR_AGGREGATED_SEQUENCES = "aggregated_sequences"
DIR_FAILED_DOMAINS = "failed_domains"


# ============================================================================
# HuggingFace Repository Configuration
# ============================================================================

REPO_ID = "compsciencelab/mdCATH"
REPO_TYPE = "dataset"


# ============================================================================
# Molecular Dynamics Configuration
# ============================================================================

# Temperatures (K) for molecular dynamics simulations
TEMPERATURES = [320, 348, 379, 413, 450]

# Protein blocks classification
PB_ORDER = {'a': 0, 'd': 1, 'h': 2, 'p': 3}  # For tie-breaking

# 3-letter to 1-letter amino acid codes
AMINO_ACID_CODES = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M',  # Selenomethionine
}


# ============================================================================
# Default Arguments
# ============================================================================

DEFAULT_PARALLEL_PROCESSES = 4
DEFAULT_PARALLEL_MAX = 4  # Cap at 4 processes by default

# CSV validation settings
MIN_ROWS_PER_RESIDUE = 25  # At least 25 trajectories (5 reps x 5 temps)
MIN_ROWS_SUMMARY = 25

# Required CSV columns
REQUIRED_PER_RESIDUE_COLUMNS = [
    'domain_id', 'replicate', 'temperature', 'residue_id', 
    'neq', 'dominant_pb', 'max_pb_freq'
]

REQUIRED_SUMMARY_COLUMNS = [
    'domain_id', 'replicate', 'temperature', 'mean_neq', 
    'median_neq', 'std_neq', 'max_neq', 'total_residues'
]

REQUIRED_METADATA_KEYS = [
    'domain_id', 'num_frames', 'num_residues', 'num_trajectories'
]
