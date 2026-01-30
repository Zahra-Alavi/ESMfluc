#!/usr/bin/env python3
"""
Aggregate protein sequences with Neq metrics across domains and temperatures.
Generates two output formats per temperature:
1. Domain-level: One row per domain with sequence and summary statistics
2. Residue-level: One row per (domain, residue_id) with individual and aggregated metrics
"""

import os
import json
import glob
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from constants import (
    TEMPERATURES, PB_ORDER, AMINO_ACID_CODES,
    get_output_base, DIR_OUTPUTS, DIR_DOWNLOADED_DATA,
    DIR_AGGREGATED_SEQUENCES, DIR_FAILED_DOMAINS, REPO_ID, REPO_TYPE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SequenceAggregator:
    """Aggregate sequence and Neq data across domains and temperatures."""
    
    TEMPERATURES = TEMPERATURES
    PB_ORDER = PB_ORDER
    
    def __init__(self, output_base: Optional[str] = None):
        """Initialize aggregator with output_base paths."""
        output_base = get_output_base(output_base)
        self.workspace_path = output_base
        self.outputs_dir = self.workspace_path / DIR_OUTPUTS
        self.downloaded_data_dir = self.workspace_path / DIR_DOWNLOADED_DATA
        self.output_dir = self.workspace_path / DIR_AGGREGATED_SEQUENCES
        self.failed_domains_dir = self.workspace_path / DIR_FAILED_DOMAINS
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create failure log
        self.failure_log_path = self.failed_domains_dir / "aggregate_failures.log"
        self.failure_handler = logging.FileHandler(self.failure_log_path)
        self.failure_handler.setLevel(logging.WARNING)
        self.failure_logger = logging.getLogger("failures")
        self.failure_logger.addHandler(self.failure_handler)
        self.failure_logger.setLevel(logging.WARNING)
    
    def extract_sequence_from_hdf5(self, domain_id: str) -> Optional[str]:
        """
        Extract protein sequence from HDF5 file using resnames.
        
        Args:
            domain_id: Domain identifier (e.g., '12asA00')
            
        Returns:
            Single-letter amino acid sequence or None if extraction fails
        """
        try:
            # Download/get from HuggingFace cache
            h5_path = hf_hub_download(
                repo_id="compsciencelab/mdCATH",
                filename=f"data/mdcath_dataset_{domain_id}.h5",
                repo_type="dataset"
            )
            h5_path = Path(h5_path)
            
        except Exception as e:
            self.failure_logger.warning(f"{domain_id}: Error downloading HDF5 file: {str(e)}")
            return None
        
        try:
            with h5py.File(h5_path, 'r') as f:
                if domain_id not in f:
                    self.failure_logger.warning(f"{domain_id}: Domain not found in HDF5 file")
                    return None
                
                domain_group = f[domain_id]
                
                # Extract sequence using resname and resid arrays at domain level
                if 'resname' not in domain_group or 'resid' not in domain_group:
                    self.failure_logger.warning(f"{domain_id}: Missing resname or resid dataset in HDF5 at domain level")
                    return None
                
                resnames = domain_group['resname'][()]
                resids = domain_group['resid'][()]
                
                if isinstance(resnames[0], bytes):
                    resnames = [r.decode('utf-8') for r in resnames]
                
                # Extract sequence by getting first atom of each unique residue ID
                sequence = self._resnames_to_sequence_by_resid(resnames, resids)
                return sequence
                    
        except Exception as e:
            self.failure_logger.warning(f"{domain_id}: Error extracting sequence from HDF5: {str(e)}")
            return None
    
    
    def _resnames_to_sequence_by_resid(self, resnames: List[str], resids: np.ndarray) -> str:
        """
        Convert 3-letter amino acid codes to single-letter sequence using residue IDs.
        
        Takes the first atom's resname for each unique residue ID.
        """
        # Get unique residue IDs in order
        unique_resids, first_indices = np.unique(resids, return_index=True)
        
        # Sort by first appearance
        sort_indices = np.argsort(first_indices)
        unique_resids = unique_resids[sort_indices]
        
        sequence = []
        for resid in unique_resids:
            # Get first atom index for this residue ID
            idx = np.where(resids == resid)[0][0]
            resname_clean = resnames[idx].strip()[:3].upper()
            code = AMINO_ACID_CODES.get(resname_clean, 'X')  # X for unknown
            sequence.append(code)
        
        return ''.join(sequence)
    
    def _resnames_to_sequence(self, resnames: List[str]) -> str:
        """Convert 3-letter amino acid codes to single-letter sequence.
        
        Since resnames has one entry per atom, we need to extract unique residues.
        Atoms are typically grouped by residue, so we take first occurrence of each unique residue.
        """
        # Extract unique residues by removing consecutive duplicates
        # (atoms are grouped by residue, so each residue appears consecutively)
        sequence = []
        prev_resname = None
        
        for resname in resnames:
            resname_clean = resname.strip()[:3].upper()
            
            # Only add when we see a new residue
            if resname_clean != prev_resname:
                code = AMINO_ACID_CODES.get(resname_clean, 'X')  # X for unknown
                sequence.append(code)
                prev_resname = resname_clean
        
        return ''.join(sequence)
    
    def load_per_residue_data(self, domain_id: str) -> Optional[pd.DataFrame]:
        """Load per_residue.csv for a domain."""
        csv_path = self.outputs_dir / domain_id / f"{domain_id}_per_residue.csv"
        
        if not csv_path.exists():
            self.failure_logger.warning(f"{domain_id}: per_residue.csv not found at {csv_path}")
            return None
        
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            self.failure_logger.warning(f"{domain_id}: Error loading per_residue.csv: {str(e)}")
            return None
    
    def validate_sequence_consistency(self, domain_id: str, df: pd.DataFrame, sequence: str) -> bool:
        """Validate that sequence is consistent across all trajectories."""
        try:
            num_residues = len(sequence)
            max_residue_id = df['residue_id'].max()
            
            if num_residues != max_residue_id:
                self.failure_logger.warning(
                    f"{domain_id}: Sequence length mismatch. "
                    f"HDF5 sequence: {num_residues} residues, "
                    f"per_residue.csv max residue_id: {max_residue_id}"
                )
                return False
            
            return True
        except Exception as e:
            self.failure_logger.warning(f"{domain_id}: Error validating sequence: {str(e)}")
            return False
    
    def aggregate_residue_metrics(self, df: pd.DataFrame, temperature: int) -> pd.DataFrame:
        """
        Aggregate per-residue metrics for a specific temperature.
        
        Returns DataFrame with columns:
        neq_rep0, neq_rep1, neq_rep2, neq_rep3, neq_rep4, neq_mean,
        pb_rep0, pb_rep1, pb_rep2, pb_rep3, pb_rep4, pb_consensus, pb_consensus_freq
        """
        # Filter for specific temperature
        temp_df = df[df['temperature'] == temperature].copy()
        
        if temp_df.empty:
            return None
        
        # Group by residue_id
        grouped = temp_df.groupby('residue_id')
        
        aggregated_rows = []
        
        for residue_id, group in grouped:
            if len(group) != 5:  # Should have 5 replicates
                logger.warning(f"Residue {residue_id}: Expected 5 replicates, found {len(group)}")
                continue
            
            row = {'residue_id': int(residue_id)}
            
            # Collect Neq values by replicate
            neq_values = []
            pb_values = []
            pb_freqs = []
            
            for _, rep_row in group.iterrows():
                rep = int(rep_row['replicate'])
                row[f'neq_rep{rep}'] = rep_row['neq']
                neq_values.append(rep_row['neq'])
                
                pb_values.append(rep_row['dominant_pb'])
                pb_freqs.append(rep_row['max_pb_freq'])
                row[f'pb_rep{rep}'] = rep_row['dominant_pb']
            
            # Calculate mean Neq
            row['neq_mean'] = np.mean(neq_values)
            
            # Consensus block (most frequent, ties broken by alphabetical order a>d>h>p)
            pb_consensus = self._get_consensus_block(pb_values)
            row['pb_consensus'] = pb_consensus
            
            # Consensus frequency (mean of frequencies for consensus block)
            consensus_freq = np.mean([
                freq for pb, freq in zip(pb_values, pb_freqs) if pb == pb_consensus
            ])
            row['pb_consensus_freq'] = consensus_freq
            
            aggregated_rows.append(row)
        
        return pd.DataFrame(aggregated_rows)
    
    def _get_consensus_block(self, blocks: List[str]) -> str:
        """Get most frequent block, with ties broken by a > d > h > p."""
        from collections import Counter
        counts = Counter(blocks)
        
        # Sort by count (descending) then by PB order (ascending)
        sorted_blocks = sorted(counts.items(), key=lambda x: (-x[1], self.PB_ORDER[x[0]]))
        return sorted_blocks[0][0]
    
    def process_domain(self, domain_id: str) -> Optional[Dict]:
        """
        Process a single domain to extract sequence and aggregate metrics.
        
        Returns:
            Dict with domain_id, sequence, and aggregated data by temperature
        """
        logger.info(f"Processing domain {domain_id}...")
        
        # Extract sequence from HDF5
        sequence = self.extract_sequence_from_hdf5(domain_id)
        if not sequence:
            return None
        
        # Load per_residue.csv
        df = self.load_per_residue_data(domain_id)
        if df is None:
            return None
        
        # Validate sequence consistency
        if not self.validate_sequence_consistency(domain_id, df, sequence):
            return None
        
        # Aggregate by temperature
        result = {
            'domain_id': domain_id,
            'sequence': sequence,
            'aggregated_data': {}
        }
        
        for temp in self.TEMPERATURES:
            agg_df = self.aggregate_residue_metrics(df, temp)
            if agg_df is not None and not agg_df.empty:
                result['aggregated_data'][temp] = agg_df
            else:
                logger.warning(f"{domain_id}: No data for temperature {temp}K")
        
        logger.info(f"✓ Successfully processed {domain_id}")
        return result
    
    def generate_output_files(self, all_results: List[Dict]):
        """Generate output CSV files for each temperature."""
        
        for temp in self.TEMPERATURES:
            logger.info(f"\nGenerating output files for {temp}K...")
            
            # Collect all domain data for this temperature
            domain_level_rows = []
            residue_level_rows = []
            
            for result in all_results:
                if temp not in result['aggregated_data']:
                    continue
                
                domain_id = result['domain_id']
                sequence = result['sequence']
                agg_df = result['aggregated_data'][temp]
                
                # Domain-level: one row per domain with per-residue Neq values as lists
                domain_row = {
                    'domain_id': domain_id,
                    'sequence': sequence,
                    'num_residues': len(sequence),
                }
                
                # Extract per-residue neq values for each replicate
                sorted_agg = agg_df.sort_values('residue_id')
                
                for rep in range(5):
                    neq_rep_col = f'neq_rep{rep}'
                    # Get sorted residue_ids and corresponding neq values
                    neq_list = sorted_agg[neq_rep_col].tolist()
                    domain_row[f'neq_values_rep_{rep}'] = neq_list
                
                # Calculate mean neq values across replicates for each residue
                mean_neq_list = sorted_agg['neq_mean'].tolist()
                domain_row['mean_neq_values'] = mean_neq_list
                
                # Dominant PB values for each replicate
                for rep in range(5):
                    pb_rep_col = f'pb_rep{rep}'
                    pb_list = sorted_agg[pb_rep_col].tolist()
                    domain_row[f'pb_values_rep_{rep}'] = pb_list
                
                # Consensus PB values and frequencies
                consensus_pb_list = sorted_agg['pb_consensus'].tolist()
                consensus_freq_list = sorted_agg['pb_consensus_freq'].tolist()
                domain_row['pb_consensus_values'] = consensus_pb_list
                domain_row['pb_consensus_frequencies'] = consensus_freq_list
                
                domain_level_rows.append(domain_row)
                
                # Residue-level: one row per (domain, residue)
                for _, res_row in sorted_agg.iterrows():
                    res_level_row = {
                        'domain_id': domain_id,
                        'sequence': sequence,
                        'residue_id': int(res_row['residue_id']),
                    }
                    
                    # Add all Neq columns
                    for rep in range(5):
                        if f'neq_rep{rep}' in res_row:
                            res_level_row[f'neq_rep{rep}'] = res_row[f'neq_rep{rep}']
                    res_level_row['neq_mean'] = res_row['neq_mean']
                    
                    # Add all PB columns
                    for rep in range(5):
                        if f'pb_rep{rep}' in res_row:
                            res_level_row[f'pb_rep{rep}'] = res_row[f'pb_rep{rep}']
                    res_level_row['pb_consensus'] = res_row['pb_consensus']
                    res_level_row['pb_consensus_freq'] = res_row['pb_consensus_freq']
                    
                    residue_level_rows.append(res_level_row)
            
            # Write domain-level file
            if domain_level_rows:
                domain_df = pd.DataFrame(domain_level_rows)
                domain_output = self.output_dir / f"sequences_{temp}K_domain_level.csv"
                domain_df.to_csv(domain_output, index=False)
                logger.info(f"✓ Written {domain_output.name} ({len(domain_df)} rows)")
            
            # Write residue-level file
            if residue_level_rows:
                residue_df = pd.DataFrame(residue_level_rows)
                residue_output = self.output_dir / f"sequences_{temp}K_residue_level.csv"
                residue_df.to_csv(residue_output, index=False)
                logger.info(f"✓ Written {residue_output.name} ({len(residue_df)} rows)")
    
    def generate_comprehensive_output(self, all_results: List[Dict]):
        """Generate a comprehensive domain-level file with all temperatures combined."""
        logger.info(f"\nGenerating comprehensive multi-temperature file...")
        
        comprehensive_rows = []
        
        for result in all_results:
            domain_id = result['domain_id']
            sequence = result['sequence']
            
            comp_row = {
                'domain_id': domain_id,
                'sequence': sequence,
                'num_residues': len(sequence),
            }
            
            # Add data for each temperature
            for temp in self.TEMPERATURES:
                if temp not in result['aggregated_data']:
                    continue
                
                agg_df = result['aggregated_data'][temp]
                sorted_agg = agg_df.sort_values('residue_id')
                
                # Add per-replicate neq values for this temperature
                for rep in range(5):
                    neq_rep_col = f'neq_rep{rep}'
                    neq_list = sorted_agg[neq_rep_col].tolist()
                    comp_row[f'{temp}K_neq_values_rep_{rep}'] = neq_list
                
                # Add mean neq values
                mean_neq_list = sorted_agg['neq_mean'].tolist()
                comp_row[f'{temp}K_mean_neq_values'] = mean_neq_list
                
                # Add per-replicate protein blocks
                for rep in range(5):
                    pb_rep_col = f'pb_rep{rep}'
                    pb_list = sorted_agg[pb_rep_col].tolist()
                    comp_row[f'{temp}K_pb_values_rep_{rep}'] = pb_list
                
                # Add consensus protein blocks
                consensus_pb_list = sorted_agg['pb_consensus'].tolist()
                consensus_freq_list = sorted_agg['pb_consensus_freq'].tolist()
                comp_row[f'{temp}K_pb_consensus_values'] = consensus_pb_list
                comp_row[f'{temp}K_pb_consensus_frequencies'] = consensus_freq_list
            
            comprehensive_rows.append(comp_row)
        
        # Write comprehensive file
        if comprehensive_rows:
            comp_df = pd.DataFrame(comprehensive_rows)
            comp_output = self.output_dir / "sequences_all_temperatures_domain_level.csv"
            comp_df.to_csv(comp_output, index=False)
            logger.info(f"✓ Written {comp_output.name} ({len(comp_df)} rows, {len(comp_df.columns)} columns)")
    
    def run(self, domain_ids: Optional[List[str]] = None):
        """
        Run aggregation for specified domains or all available domains.
        
        Args:
            domain_ids: List of domain IDs to process. If None, process all.
        """
        # Discover domains
        if domain_ids is None:
            domain_dirs = [d.name for d in self.outputs_dir.iterdir() if d.is_dir()]
            domain_ids = sorted(domain_dirs)
        
        if not domain_ids:
            logger.error(f"No domains found in {self.outputs_dir}")
            return
        
        logger.info(f"Found {len(domain_ids)} domain(s) to process")
        
        # Process each domain
        results = []
        for domain_id in tqdm(domain_ids, desc="Processing domains"):
            try:
                result = self.process_domain(domain_id)
                if result:
                    results.append(result)
            except Exception as e:
                self.failure_logger.error(f"{domain_id}: Unexpected error: {str(e)}")
                logger.error(f"{domain_id}: {str(e)}")
        
        # Generate output files
        if results:
            self.generate_output_files(results)
            self.generate_comprehensive_output(results)
            logger.info(f"\n✓ Successfully processed {len(results)}/{len(domain_ids)} domains")
            logger.info(f"Output files written to {self.output_dir}")
        else:
            logger.error("No domains were successfully processed")
        
        # Summary
        if self.failure_log_path.stat().st_size > 0:
            logger.info(f"\nFailure log: {self.failure_log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate protein sequences with Neq metrics across domains and temperatures"
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="data/mdcath",
        help="Base directory for all outputs (default: data/mdcath relative to workspace)"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Specific domains to process (default: all)"
    )
    
    args = parser.parse_args()
    
    aggregator = SequenceAggregator(args.output_base)
    aggregator.run(args.domains)


if __name__ == "__main__":
    main()
