# mdCATH Neq Calculation Pipeline

A modular pipeline for downloading mdCATH protein structure data, calculating Neq (equivalent number of protein blocks) values, extracting protein sequences, and aggregating results across domains and temperatures.

## Overview

This pipeline consists of five main stages:

1. **Download** (`data_download.py`) - Download mdCATH HDF5 files from HuggingFace
2. **Extract** (`extract_trajectories.py`) - Extract trajectory data from HDF5 files
3. **Calculate Neq** (`calculate_neq.py`) - Calculate Neq values per residue per trajectory
4. **Export** (`export_results.py`) - Export results to per-domain CSV files with metadata
5. **Aggregate** (`aggregate_sequences.py`) - Aggregate sequences and Neq metrics across all domains

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Single Domain (Testing)

```bash
# Download, process, and aggregate single domain
python run_pipeline.py --domain 12asA00

# Or step by step:
python data_download.py --domain 12asA00
python extract_trajectories.py --domain 12asA00
python calculate_neq.py --domain 12asA00
python export_results.py --domain 12asA00
python aggregate_sequences.py . --domains 12asA00
```

### Multiple Specific Domains

```bash
# Download, process, and aggregate multiple domains (with parallelization)
python run_pipeline.py --domains 12asA00 1mba00 3aab00 --parallel 4
```

### Full Pipeline (All Domains)

```bash
# Full pipeline with parallelization (includes aggregation)
python run_pipeline.py --all --parallel 4

# Resume from interruption (includes aggregation)
python run_pipeline.py --all --parallel 4 --resume
```

## Usage

### Data Download

```bash
# Single domain
python data_download.py --domain 12asA00

# Multiple specific domains
python data_download.py --domains 12asA00 1mba00 3aab00

# All domains
python data_download.py --all

# Resume download (skip already downloaded)
python data_download.py --all --resume
```

**Output:** Downloaded HDF5 files stored in HuggingFace cache, paths recorded in `downloaded_data/downloaded_files.txt`

### Extract Trajectories

```bash
# From single domain
python extract_trajectories.py --domain 12asA00

# From list of domains
python extract_trajectories.py --input-list domains.txt

# With parallelization
python extract_trajectories.py --input-list domains.txt --parallel 4

# Keep intermediate PDB files (default: delete)
python extract_trajectories.py --input-list domains.txt --keep-intermediates

# Resume incomplete extractions
python extract_trajectories.py --input-list domains.txt --resume
```

**Output:** Extracted trajectories in `extracted_data/{domain_id}/` with metadata

### Calculate Neq Values

```bash
# Single domain
python calculate_neq.py --domain 12asA00

# Multiple domains
python calculate_neq.py --input-list domains.txt

# Parallel processing
python calculate_neq.py --input-list domains.txt --parallel 4

# Resume from failures
python calculate_neq.py --input-list domains.txt --resume
```

**Output:** Per-domain results in `neq_results/{domain_id}/` including:

- `{domain_id}_per_residue.csv` - Per-residue Neq values for each trajectory
- `{domain_id}_summary.csv` - Summary statistics per trajectory
- `metadata.json` - Processing metadata

### Export Results

```bash
# Single domain
python export_results.py --domain 12asA00

# Multiple domains
python export_results.py --input-list domains.txt

# All processed domains
python export_results.py --all

# Verify integrity during export
python export_results.py --all --verify
```

**Output:** Final results in `outputs/{domain_id}/` with CSV files and README

### Aggregate Sequences and Metrics

**Combine results across all domains and temperatures into comprehensive files with protein sequences and aggregated Neq metrics.**

```bash
# Aggregate all processed domains (auto-discovers from outputs/)
python aggregate_sequences.py .

# Aggregate specific domains only
python aggregate_sequences.py . --domains 12asA00 1mba00 3aab00
```

**Output:** Two output formats per temperature in `aggregated_sequences/`:

1. **Domain-level** (`sequences_{TEMP}K_domain_level.csv`)
    - One row per domain
    - Full protein sequence
    - Per-residue Neq values for all 5 replicates + mean
    - Per-residue dominant protein blocks for all 5 replicates + consensus + consensus frequency

2. **Residue-level** (`sequences_{TEMP}K_residue_level.csv`)
    - One row per (domain, residue)
    - Domain ID and full protein sequence on each row
    - Residue position
    - Individual Neq values per replicate (neq_rep0-4) + mean
    - Individual protein blocks per replicate (pb_rep0-4) + consensus + consensus frequency

**Example files generated:**

- `sequences_320K_domain_level.csv` - 320K temperature, domain aggregation
- `sequences_320K_residue_level.csv` - 320K temperature, residue aggregation
- `sequences_348K_domain_level.csv` - 348K temperature, domain aggregation
- ... (continues for all 5 temperatures: 348K, 379K, 413K, 450K)
- **`sequences_all_temperatures_domain_level.csv`** - Comprehensive file with all temperatures combined at domain level

**Comprehensive Multi-Temperature File:**

The `sequences_all_temperatures_domain_level.csv` file contains all temperature data in a single file for easy cross-temperature comparisons:

- One row per domain
- Full protein sequence
- For each temperature (320K, 348K, 379K, 413K, 450K):
    - `{TEMP}K_neq_values_rep_0` through `{TEMP}K_neq_values_rep_4` - Per-replicate Neq values per residue
    - `{TEMP}K_mean_neq_values` - Mean Neq values across replicates
    - `{TEMP}K_pb_values_rep_0` through `{TEMP}K_pb_values_rep_4` - Per-replicate protein blocks per residue
    - `{TEMP}K_pb_consensus_values` - Consensus protein blocks per residue
    - `{TEMP}K_pb_consensus_frequencies` - Frequencies of consensus blocks

**Total output files for 5 temperatures:** 11 files

- 10 files organized by temperature (2 formats × 5 temperatures)
- 1 comprehensive file combining all temperatures at domain level

**Failure handling:** Failed domains are logged to `failed_domains/aggregate_failures.log` with detailed error messages. Successfully processed domains are included in the output files.

### Full Pipeline

```bash
# Test single domain
python run_pipeline.py --domain 12asA00
python aggregate_sequences.py . --domains 12asA00

# Full pipeline (includes aggregation)
python run_pipeline.py --all --parallel 4

# Resume with specific skip options
python run_pipeline.py --all --parallel 4 --resume --skip-download --keep-intermediates

# Skip aggregation if you don't want it
python run_pipeline.py --all --parallel 4 --skip-aggregate
```

**Options for run_pipeline.py:**

- `--domain`, `--domains`, `--all`: Which domains to process
- `--parallel N`: Number of parallel processes (default: 1)
- `--resume`: Skip completed steps/domains
- `--skip-download`: Skip download step
- `--skip-extract`: Skip extraction step
- `--skip-neq`: Skip Neq calculation step
- `--skip-export`: Skip export step
- `--skip-aggregate`: Skip aggregation step (default: aggregation is included)
- `--keep-intermediates`: Keep extracted trajectory files
- `--keep-intermediates`: Keep extracted trajectory files
- `--skip-download`: Skip download step
- `--skip-extract`: Skip extraction step
- `--skip-neq`: Skip Neq calculation step
- `--skip-export`: Skip export step

**Options for aggregate_sequences.py:**

- `.`: Workspace path (current directory)
- `--domains ID1 ID2 ...`: Specific domains to aggregate (optional, default: all)

## Output Structure

```
./downloaded_data/          # Downloaded HDF5 file paths
  downloaded_files.txt      # List of downloaded files

./outputs/                  # Final domain-level results
  {domain_id}/
    {domain_id}_per_residue.csv    # Per-residue Neq values (25 trajectories × residues)
    {domain_id}_summary.csv        # Summary statistics (one per trajectory)
    metadata.json                  # Processing metadata
    README.md                      # Domain results summary

./aggregated_sequences/     # Aggregated results across all domains
  sequences_320K_domain_level.csv            # All domains, domain level (320K)
  sequences_320K_residue_level.csv           # All domains, residue level (320K)
  sequences_348K_domain_level.csv            # 348K temperature
  sequences_348K_residue_level.csv
  sequences_379K_domain_level.csv            # 379K temperature
  sequences_379K_residue_level.csv
  sequences_413K_domain_level.csv            # 413K temperature
  sequences_413K_residue_level.csv
  sequences_450K_domain_level.csv            # 450K temperature
  sequences_450K_residue_level.csv
  sequences_all_temperatures_domain_level.csv  # All temps combined at domain level

./failed_domains/           # Error logs from pipeline
  aggregate_failures.log    # Detailed errors from aggregation
  {domain_id}_error.log     # Detailed error messages per domain
```

## CSV File Formats

### Per-Residue CSV (outputs/)

Columns: `domain_id`, `replicate`, `temperature`, `residue_id`, `neq`, `dominant_pb`, `max_pb_freq`, `num_unique_pbs`

Each row represents one residue in one trajectory (25 trajectories × residues rows per domain)

### Summary CSV (outputs/)

Columns: `domain_id`, `replicate`, `temperature`, `mean_neq`, `median_neq`, `std_neq`, `max_neq`, `total_residues`

One row per trajectory (25 rows per domain for 5 temperatures × 5 replicates)

### Domain-Level Aggregation CSV (aggregated_sequences/)

Columns per temperature:

- `domain_id` - Domain identifier
- `sequence` - Full protein sequence (single-letter amino acid codes)
- `num_residues` - Number of residues
- `neq_values_rep_0` to `neq_values_rep_4` - Neq values per residue for each replicate (list)
- `mean_neq_values` - Mean Neq values across replicates (list)
- `pb_values_rep_0` to `pb_values_rep_4` - Dominant protein blocks per residue for each replicate (list)
- `pb_consensus_values` - Consensus protein block per residue (list)
- `pb_consensus_frequencies` - Frequency of consensus block (list)

**Example:**

```
domain_id,sequence,num_residues,neq_values_rep_0,neq_values_rep_1,...,mean_neq_values,...
12asA00,AYIAKQRQ...,327,"[2.15, 2.13, 2.14, ...]","[2.16, 2.14, ...]",...,"[2.15, 2.14, ...]",...
```

### Residue-Level Aggregation CSV (aggregated_sequences/)

Columns per temperature:

- `domain_id` - Domain identifier
- `sequence` - Full protein sequence
- `residue_id` - Residue position (1-indexed)
- `neq_rep0` to `neq_rep4` - Individual Neq values per replicate
- `neq_mean` - Mean Neq across replicates
- `pb_rep0` to `pb_rep4` - Individual protein blocks per replicate (a/d/h/p)
- `pb_consensus` - Consensus protein block (most frequent)
- `pb_consensus_freq` - Frequency of consensus block

**Example:**

```
domain_id,sequence,residue_id,neq_rep0,neq_rep1,neq_rep2,neq_rep3,neq_rep4,neq_mean,pb_rep0,pb_rep1,pb_rep2,pb_rep3,pb_rep4,pb_consensus,pb_consensus_freq
12asA00,AYIAKQRQ...,1,2.15,2.16,2.14,2.13,2.12,2.14,p,p,p,p,p,p,0.95
12asA00,AYIAKQRQ...,2,2.13,2.14,2.12,2.11,2.10,2.12,h,h,h,h,h,h,0.92
```

## Error Handling

Failed domains during each stage are logged:

```
./failed_domains/
  failed_list.txt              # List of domain IDs from pipeline
  aggregate_failures.log       # Aggregation-specific failures
  {domain_id}_error.log        # Detailed error traceback per domain
```

Use `--resume` flag to retry failed domains in the pipeline.

## Performance Notes

- **Parallelization**: Default limit is `min(4, cpu_count)` to avoid memory issues
- **CPU vs GPU**: All processing is CPU-based, no GPU required
- **Memory**: Each trajectory can be large; sequential processing is default
- **Time**: Full pipeline for 5,398 domains with `--parallel 4` typically takes several days
- **Aggregation**: Single-threaded, processes all domains sequentially (~minutes for complete dataset)

## Validation

Resume functionality validates:

- CSV file existence and non-empty status
- Correct number of rows and columns
- Metadata file completeness

Use `export_results.py --verify` to check results before aggregation.

## Troubleshooting

### Download fails

- Check internet connection
- Verify HuggingFace repository access
- Use `--resume` to skip already downloaded domains

### Extraction fails

- Verify HDF5 file structure with `h5py`
- Check disk space for intermediate files
- Examine `failed_domains/{domain_id}_error.log`

### Neq calculation fails

- Insufficient memory: reduce `--parallel` value
- Missing dependencies: `pip install -r requirements.txt`
- Check `failed_domains/{domain_id}_error.log` for details

### Aggregation fails

- Domain not yet processed: run pipeline first
- Check `failed_domains/aggregate_failures.log` for details
- Verify `neq_results/{domain_id}/` and `extracted_data/{domain_id}/` exist

### Export validation fails

- Re-run Neq calculation for failed domain
- Check CSV file integrity with pandas
- Verify expected row/column counts

## Dependencies

See `requirements.txt` for all dependencies. Key libraries:

- `huggingface-hub`: Download from HuggingFace
- `h5py`: Read HDF5 files
- `pandas`: CSV handling and aggregation
- `numpy`: Numerical computations
- `portalocker`: Thread-safe file operations
- `tqdm`: Progress bars

## Citation

If you use this pipeline, please cite:

- mdCATH: Mirarchi, A., Giorgino, T. & De Fabritiis, G. (2024). Scientific Data 11:1299
- pbxplore: Barnoud J. et al. (2017). PeerJ 5:e4013
