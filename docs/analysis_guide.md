# Reproducing the analyses

Run commands from the repository root in the Python 3.11 environment described in the README. Write reruns to new directories rather than overwriting the shipped results. `results/README.md` and its CSV index connect the paper's figures, claims and ten supplementary tables (S1–S10) to their saved sources.

Weights, trajectories, raw server exports and large run arrays are not included. A published model deposit and upstream inputs are needed for a complete rerun. Some saved parameter files store relative paths; `provenance/original_checksums.json` lists the SHA-256 of each file as originally generated.

## 1. Fixed ATLAS inputs

`data/atlas_grouped_v1/split_manifest_grouped_v1.csv` fixes identifiers, sequence hashes, split membership and union groups. There are 967/208/208 train/validation/test proteins and 223,573/47,875/47,751 residues. All 1,383 identifiers belong to 700 union groups. Input and output checksums are in `split_summary_grouped_v1.json`; training seeds never change membership.

Retrieve the per-protein ATLAS Neq tables and reconstruct the split CSVs/FASTAs. This command can fetch missing tables from their ATLAS analysis archives, recover sequences, check their frozen hashes, and average the three replicate-specific targets:

```bash
python -m esmfluc.datasets.reconstruct_atlas_split --table-dir external_inputs/atlas_neq --fetch --output-dir data/atlas_grouped_v1
```

Missing tables, changed sequences and invalid targets stop reconstruction. Compare emitted CSV hashes with the split summary before claiming byte-identical reproduction. `--names` supports small-input checks only.

For a grouping audit, obtain the original ATLAS metadata/target CSVs and ECOD develop285. Pinned ECOD URL: `http://prodata.swmed.edu/ecod/distributions/ecod.develop285.domains.txt`; SHA-256: `73da73a314e59fda27f734909f30f3df6d852751358047cd74231a4d56512299`. Save it under `external_inputs/` and verify its checksum. The retained parser specifies the source CSVs, duplicate-target tables and MMseqs2 binary:

```bash
python -m esmfluc.datasets.create_atlas_grouped_splits --help
```

Grouping used 30% sequence identity, 80% coverage of both sequences (`--cov-mode 0`) and split seed 42. These differ from the mdCATH contamination-filter settings. Use the frozen membership manifest for training.

Figure 1 is a manually assembled schematic. Its distribution input is supplied as `data/atlas_grouped_v1/training_target_distribution.csv`; no single script generates the complete figure.

## 2. Train or restore the models

The seven configurations and saved arguments are in `provenance/model_release/manifest.json`. This copy is for reference; restoring weights uses the manifest distributed with the Zenodo model record.

Six conditions use three bidirectional LSTM layers, 512 hidden units per direction, dropout 0.3, single-head self-attention and a binary linear classifier. The frozen ESM2 linear baseline bypasses the BiLSTM and task-specific attention. Training uses focal loss (gamma 2), AdamW (1e-5; weight decay 0.01), batch size 2, mixed precision, at most 80 epochs, ReduceLROnPlateau (factor 0.1; patience 3), and early stopping (patience 5; minimum improvement 0.001). Each condition uses seeds 1, 2 and 3.

The full training workflow validates fixed inputs and records checkpoints, predictions, metrics and a run manifest. Configure its output name and GPU identifiers for your machine before running; this starts expensive training:

```bash
bash workflows/run_training_pipeline.sh
python -m esmfluc.evaluation.analyze_seed_variance --help
```

Table S3 describes configurations. Table S6 and Figure 2a,b use pooled ATLAS AUROC, AUPRC and Spearman from `results/atlas_performance/pooled_residue_performance_{by_seed,across_seeds}.csv`. Protein-macro summaries are separate and must not be substituted for pooled metrics.

The ordinary test suite does not require model downloads. Its optional real-checkpoint test can use a separate read-only run directory through `ESMFLUC_TEST_RUNS_DIR`; do not copy weights into the public source tree.

## 3. Regenerate contribution arrays

Restore each attention checkpoint to `best_model.pth` and supply the reconstructed split CSVs. The extractor loads a checkpoint once and streams the requested splits:

```bash
python -m esmfluc.inference.extract_all_split_signed_contributions --checkpoint models/seed_1.pth --condition esm3_top28_bilstm_attn --seed 1 --esm_model esm3_sm_open_v1 --is_esm3 --split test data/atlas_grouped_v1/test_grouped_v1.csv --output_dir runs/esm3_top28_bilstm_attn/seed_1/all_split_signed_contributions
```

Repeat for all three seeds, all six attention conditions and all splits. For a full batch, configure `RESULT_ROOT` and `MANIFEST` for the retained wrapper. It requires the run manifest and copied split CSVs from training:

```bash
bash workflows/run_all_split_signed_contributions_v2.sh
python -m esmfluc.inference.audit_all_split_signed_contributions --help
```

The audit checks exact row-sum-plus-bias reconstruction. The seed-average operation adds fields atomically to its input files, so use regenerated working copies, not archival source files:

```bash
python -m esmfluc.analysis.bands add-seed-average --help
```

## 4. Bands and seed stability

Provide TSV manifests with `condition`, `seed`, `split`, `json_gz`, pointing to the regenerated profiles. Run the raw detector separately on each seed and the three-seed mean:

```bash
python -m esmfluc.analysis.bands extract-bands --manifest_tsv runs/contribution_profiles.tsv --output_dir runs/bands/per_seed --influence_field signed_column_influence --apex_method raw_mad_amplitude --amplitude_mad 2 --support_method half_intensity_merge --terminal_exclusion 0 --terminal_exclusion_fraction 0
python -m esmfluc.analysis.bands extract-bands --manifest_tsv runs/mean_profiles.tsv --output_dir runs/bands/mean --influence_field seed_averaged_signed_column_influence --apex_method raw_mad_amplitude --amplitude_mad 2 --support_method half_intensity_merge --terminal_exclusion 0 --terminal_exclusion_fraction 0
python -m esmfluc.analysis.bands seed-reproducibility --bands_csv runs/bands/per_seed/signed_bands.csv --protein_summary_csv runs/bands/per_seed/signed_band_protein_summary.csv --mean_bands_csv runs/bands/mean/signed_bands.csv --output_dir runs/bands/reproducibility --matching_method interval_iou --min_interval_iou 0.5 --min_seeds 2 --n_block_shifts 1000 --random_seed 123
```

The amplitude threshold is two MAD-based scale units; intervals follow the contiguous same-sign half-height rule and overlapping intervals merge. One-to-one matching is within condition/protein/split/sign, requiring IoU at least 0.5. The saved test catalog contains 11,039 primary bands supported by at least two seeds. Protein summaries include zero-band proteins and supply necessary denominators.

## 5. Local properties, attention controls and structure

Submit the reconstructed split FASTAs to the NetSurfP-3.0 **web server**, retaining submission dates and raw exports privately. Validate every identifier, sequence and annotation-array length before joining. Expected record counts: 967/208/208 ATLAS and 989 strict mdCATH. A changed server version may not reproduce the reported values; no standalone NetSurfP program is bundled.

The annotation/enrichment parsers specify the stable-band catalog, mean-profile protein summary, split files and joined residue annotations. Retain group-level inference, recorded matching settings and the correct BH family:

```bash
python -m esmfluc.analysis.bands annotate-biophysics --help
python -m esmfluc.analysis.bands biophysical-enrichment --help
python -m esmfluc.analysis.bands audit-biophysics --help
```

`workflows/run_signed_band_biophysical_pipeline.sh` runs these three steps with the settings used for the paper; set `BANDS_CSV`, `PROTEIN_SUMMARY_CSV`, `OUTPUT_ROOT` and, for MD strain, `STRAIN_ROOT`.

Figure 4a–d uses `results/local_properties/enrichment_with_test_strain/headline_union_group_inference.csv`. RSA compares same-protein Q3/Neq-matched peaks and controls; the other primary properties additionally match RSA. Measurements are at peaks, not averaged over bands.

Strain came from [MDStrainMapper](https://github.com/Zahra-Alavi/MDStrainMapper) v1.0 (commit `bbe8efc`). See `provenance/mdstrainmapper.json`. Strain is restricted to test proteins here.

Uniform and shifted attention controls use their recorded profile definitions and the same band settings. Component localization is descriptive supporting context, not an enrichment test:

```bash
python -m esmfluc.analysis.bands build-uniform-control --help
python -m esmfluc.analysis.bands compare-uniform-bands --help
python -m esmfluc.analysis.bands model-mechanism --help
python -m esmfluc.analysis.bands apex-component-localization --bands-csv runs/bands/reproducibility/stable_signed_bands.csv --profiles-manifest-tsv runs/component_profiles.tsv --output-dir runs/component_localization --splits test
```

The last command produces Table S4 from the compact three-seed component-profile manifest, comparing stable apices with other same-sign influence extrema and all same-sign interior positions. It never assumes that the mean influence equals the product of separately averaged factors.

Build alignment-audited PDB contact files before structural analysis. Missing coordinates remain missing:

```bash
python -m esmfluc.inference.build_contact_maps_from_pdb --help
python -m esmfluc.analysis.bands apex-structure --help
python -m esmfluc.analysis.bands band-interval-structure --help
```

Figure 4e uses the primary contact-count effects; RSA-matched effects and interval sensitivities are separate outputs. Final figure layouts were assembled manually and are not included. The result index identifies source tables and plotters without claiming a script generates an entire manually assembled figure.

## 6. Query receivers

Run receiver analysis after source bands, mechanism categories, contribution arrays and query annotations exist. The wrapper can add audited structural/water features; set its `RECEIVER_*` inputs to newly generated files first: Training writes to `results/<RESULT_SET>/`; to analyze a new training run, set `RESULT_ROOT=results/<RESULT_SET>` for this wrapper and for `run_all_split_signed_contributions_v2.sh`, so both read the same run. `RECEIVER_MANIFEST_TSV` and `RECEIVER_CONTACT_ROOT` override individual inputs.

```bash
python -m esmfluc.analysis.bands query-receivers --help
python -m esmfluc.analysis.bands build-query-structure --help
bash workflows/run_query_receivers.sh
```

The wrapper also needs the ECOD develop285 annotation table, which is not distributed. Generate it with `create_atlas_grouped_splits` (Section 1) using a separate `--output-dir`, so the shipped split files are not overwritten, and point the wrapper to it with `RECEIVER_ECOD_CSV=<output-dir>/ecod_v285_annotations.csv`.

Table S5 compares the distance/Q8 baseline with the next stage adding query biophysics, within the recorded fixed eligibility cohorts. Long-range fractions use combined-class source bands and protein-averaged test summaries. Different eligibility cohorts are not interchangeable.

## 7. Target sensitivity and benchmarking

Tables S1–S2 require saved residue predictions and replicate ATLAS Protein Block assignments from the analysis archives, not just mean Neq values:

```bash
python -m esmfluc.analysis.endpoint.analyze_neq_pb_reliability --help
```

Outputs include stricter endpoints, replicate-consensus performance and alignment/target-reconstruction audits. The public pipeline does not generate manuscript prose.

The frozen mdCATH cohort has 989 domains; 985 have length-aligned Neq targets. Its filter removes PDB-code overlap, then MMseqs2 matches at 30% identity and 50% **query (ATLAS)** coverage (`--cov-mode 2`). `filter_provenance.json` and cluster assignments retain the cohort/grouping record. Obtain source trajectories from mdCATH if rerunning cohort construction; do not silently substitute a current database release.

mdCATH Neq targets: for each domain in `data/mdcath/domains.txt`, Protein Blocks are assigned with PBxplore to every saved frame of each replica. Neq is computed per replica and averaged over the five replicas at each temperature. The first and last two residues have no Protein Block and get Neq = 1. Domains are then split by MMseqs2 cluster (seed 42); the 1,069-domain test set at 320 K is the cohort before ATLAS filtering. This downloads every mdCATH trajectory file:

```bash
pip install -e ".[mdcath]"
python -m esmfluc.datasets.compute_mdcath_neq --data_root data/mdcath
MMSEQS_BIN=/path/to/mmseqs python -m esmfluc.datasets.split_mdcath_clusters --data_root data/mdcath
python -m esmfluc.datasets.split_mdcath_by_temperature
```

Regenerate PEGASUS models/predictions and DynaMine scores from their providers. The PEGASUS workflow records its download URL and Docker image. The DynaMine client sends sequences to Bio2Byte; inspect inputs before submitting. Remote RMSF extraction can transfer large volumes and requires explicit confirmation:

```bash
python -m esmfluc.benchmarks.benchmark_pegasus_mdcath --help
python -m esmfluc.datasets.fetch_dynamine_mdcath --help
python -m esmfluc.datasets.extract_mdcath_rmsf --help
python -m esmfluc.benchmarks.benchmark_unified_mdcath --help
python -m esmfluc.benchmarks.benchmark_atlas_mdcath_missing_metrics --help
```

Tables S7–S8 and Figure 2c,d use `unified_method_summary.csv`. Table S9 uses `results/benchmark_unified_mdcath/unified_paired_deltas.csv`, with paired-domain differences and cluster-bootstrap intervals. The metric-completion module computes the reported mdCATH AUROC tables. Table S10 repeats the Neq benchmark with `--exclude-ends 2` (without `--rmsf-csv`), written to `results/benchmark_unified_mdcath_ends_excluded/`.

## 8. Guanylate kinase

See `examples/gk_wt/FIGURE.md` for Figure 3. Three-seed contribution matrices and saved influence bands are combined with a manually rendered view of PDB 1ZNX chain A. The structure illustrates the sequence-based result; it is not a prediction input or causal validation.
