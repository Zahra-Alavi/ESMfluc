# Reproducing the analyses

Run commands from the repository root in the Python 3.11 environment from the README, and write reruns to new directories so the shipped results stay intact. `results/README.md` lists the saved file behind each figure, table and number in the paper.

Weights, trajectories, raw server exports and large run arrays are not included; a full rerun needs the Zenodo models and the upstream data. `provenance/original_checksums.json` lists the SHA-256 of each file as originally generated.

## 1. Fixed ATLAS inputs

`data/atlas_grouped_v1/split_manifest_grouped_v1.csv` fixes identifiers, sequence hashes, split membership and union groups. There are 967/208/208 train/validation/test proteins and 223,573/47,875/47,751 residues. All 1,383 identifiers belong to 700 union groups. Input and output checksums are in `split_summary_grouped_v1.json`; training seeds never change membership.

Retrieve the per-protein ATLAS Neq tables and reconstruct the split CSVs/FASTAs. This command fetches missing tables from the ATLAS analysis archives, checks the sequence hashes and averages the three replicate targets:

```bash
python -m esmfluc.datasets.reconstruct_atlas_split --table-dir external_inputs/atlas_neq --fetch --output-dir data/atlas_grouped_v1
```

Reconstruction stops on missing tables, changed sequences or invalid targets. Compare the output CSV hashes with the split summary. `--names` runs a subset for quick checks.

To rebuild the grouping, download the original ATLAS metadata and target CSVs and ECOD develop285 (`http://prodata.swmed.edu/ecod/distributions/ecod.develop285.domains.txt`, SHA-256 `73da73a314e59fda27f734909f30f3df6d852751358047cd74231a4d56512299`) into `external_inputs/`. The command's help lists its inputs:

```bash
python -m esmfluc.datasets.create_atlas_grouped_splits --help
```

Grouping used 30% sequence identity, 80% coverage of both sequences (`--cov-mode 0`) and split seed 42; the mdCATH filter uses different settings. Train on the frozen split manifest.

Figure 1 is a schematic assembled by hand; its distribution panel uses `data/atlas_grouped_v1/training_target_distribution.csv`.

## 2. Train or restore the models

`provenance/model_release/manifest.json` lists the seven configurations and their saved arguments. Restoring weights uses the copy on Zenodo.

Six conditions use three bidirectional LSTM layers, 512 hidden units per direction, dropout 0.3, single-head self-attention and a binary linear classifier. The frozen ESM2 linear baseline bypasses the BiLSTM and task-specific attention. Training uses focal loss (gamma 2), AdamW (1e-5; weight decay 0.01), batch size 2, mixed precision, at most 80 epochs, ReduceLROnPlateau (factor 0.1; patience 3), and early stopping (patience 5; minimum improvement 0.001). Each condition uses seeds 1, 2 and 3.

The training workflow checks the fixed inputs and saves checkpoints, predictions, metrics and a run manifest. Set the output name and GPU IDs first; training is expensive:

```bash
bash workflows/run_training_pipeline.sh
python -m esmfluc.evaluation.analyze_seed_variance --help
```

Table S3 lists the configurations. Table S6 and Figure 2a,b use pooled ATLAS AUROC, AUPRC and Spearman from `results/atlas_performance/pooled_residue_performance_{by_seed,across_seeds}.csv`; protein-averaged summaries are separate files.

The tests do not need model downloads. An optional test runs on real checkpoints when `ESMFLUC_TEST_RUNS_DIR` points to a run directory.

## 3. Regenerate contribution arrays

Restore each attention checkpoint and pass the reconstructed split CSVs:

```bash
python -m esmfluc.inference.extract_all_split_signed_contributions --checkpoint models/seed_1.pth --condition esm3_top28_bilstm_attn --seed 1 --esm_model esm3_sm_open_v1 --is_esm3 --split test data/atlas_grouped_v1/test_grouped_v1.csv --output_dir runs/esm3_top28_bilstm_attn/seed_1/all_split_signed_contributions
```

Repeat for each seed, attention condition and split. For a full batch, set `RESULT_ROOT` and `MANIFEST` in the wrapper, which uses the run manifest and split CSVs saved by training:

```bash
bash workflows/run_all_split_signed_contributions_v2.sh
python -m esmfluc.inference.audit_all_split_signed_contributions --help
```

The audit checks that row sums plus bias reconstruct the logits exactly. Seed averaging writes new fields into its input files, so run it on regenerated copies:

```bash
python -m esmfluc.analysis.bands add-seed-average --help
```

## 4. Bands and seed stability

Make TSV manifests with `condition`, `seed`, `split` and `json_gz` for the regenerated profiles. Run the detector on each seed and on the three-seed mean, then match bands across seeds:

```bash
python -m esmfluc.analysis.bands extract-bands --manifest_tsv runs/contribution_profiles.tsv --output_dir runs/bands/per_seed --influence_field signed_column_influence --apex_method raw_mad_amplitude --amplitude_mad 2 --support_method half_intensity_merge --terminal_exclusion 0 --terminal_exclusion_fraction 0
python -m esmfluc.analysis.bands extract-bands --manifest_tsv runs/mean_profiles.tsv --output_dir runs/bands/mean --influence_field seed_averaged_signed_column_influence --apex_method raw_mad_amplitude --amplitude_mad 2 --support_method half_intensity_merge --terminal_exclusion 0 --terminal_exclusion_fraction 0
python -m esmfluc.analysis.bands seed-reproducibility --bands_csv runs/bands/per_seed/signed_bands.csv --protein_summary_csv runs/bands/per_seed/signed_band_protein_summary.csv --mean_bands_csv runs/bands/mean/signed_bands.csv --output_dir runs/bands/reproducibility --matching_method interval_iou --min_interval_iou 0.5 --min_seeds 2 --n_block_shifts 1000 --random_seed 123
```

Peaks must exceed two MAD units; each band extends over the contiguous same-sign positions above half the peak height, and overlapping bands merge. Bands are matched one-to-one within condition, protein, split and sign, at IoU of at least 0.5. The saved test catalog has 11,039 bands found by at least two seeds. Protein summaries include proteins with no bands.

## 5. Local properties, attention controls and structure

Submit the split FASTAs to the NetSurfP-3.0 web server and keep the submission dates and raw exports. Check identifiers, sequences and array lengths before joining. Expected records: 967/208/208 ATLAS and 989 mdCATH. A newer server version may give different values.

Then annotate the bands, test for enrichment and audit the results. Each command's help lists its inputs:

```bash
python -m esmfluc.analysis.bands annotate-biophysics --help
python -m esmfluc.analysis.bands biophysical-enrichment --help
python -m esmfluc.analysis.bands audit-biophysics --help
```

`workflows/run_signed_band_biophysical_pipeline.sh` runs these three steps with the settings used for the paper; set `BANDS_CSV`, `PROTEIN_SUMMARY_CSV`, `OUTPUT_ROOT` and, for MD strain, `STRAIN_ROOT`.

Figure 4a–d uses `results/local_properties/enrichment_with_test_strain/headline_union_group_inference.csv`. RSA compares same-protein Q3/Neq-matched peaks and controls; the other primary properties additionally match RSA. Values are measured at peaks.

Strain values come from [MDStrainMapper](https://github.com/Zahra-Alavi/MDStrainMapper) v1.0 (commit `bbe8efc`; `provenance/mdstrainmapper.json`) and cover test proteins only.

The uniform and shifted attention controls use the same band settings:

```bash
python -m esmfluc.analysis.bands build-uniform-control --help
python -m esmfluc.analysis.bands compare-uniform-bands --help
python -m esmfluc.analysis.bands model-mechanism --help
python -m esmfluc.analysis.bands apex-component-localization --bands-csv runs/bands/reproducibility/stable_signed_bands.csv --profiles-manifest-tsv runs/component_profiles.tsv --output-dir runs/component_localization --splits test
```

The last command produces Table S4, comparing stable peaks with other same-sign extrema and with all same-sign interior positions.

Build PDB contact files, with the sequence alignment checked, before the structural analysis:

```bash
python -m esmfluc.inference.build_contact_maps_from_pdb --help
python -m esmfluc.analysis.bands apex-structure --help
python -m esmfluc.analysis.bands band-interval-structure --help
```

Figure 4e uses the primary contact-count effects; RSA-matched effects and interval sensitivities are separate outputs. Figure layouts were assembled by hand and are not included.

## 6. Query receivers

Run the receiver analysis once bands, mechanism categories, contribution arrays and query annotations exist. Set the wrapper's `RECEIVER_*` inputs to your new files; it can also add structural and water features. Training writes to `results/<RESULT_SET>/`, so set `RESULT_ROOT=results/<RESULT_SET>` for this wrapper and for `run_all_split_signed_contributions_v2.sh` to read the same run. `RECEIVER_MANIFEST_TSV` and `RECEIVER_CONTACT_ROOT` override single inputs.

```bash
python -m esmfluc.analysis.bands query-receivers --help
python -m esmfluc.analysis.bands build-query-structure --help
bash workflows/run_query_receivers.sh
```

The wrapper also needs the ECOD develop285 annotation table, which is not distributed. Generate it with `create_atlas_grouped_splits` (Section 1) using a separate `--output-dir`, so the shipped split files are not overwritten, and point the wrapper to it with `RECEIVER_ECOD_CSV=<output-dir>/ecod_v285_annotations.csv`.

Table S5 compares the distance/Q8 baseline with the stage that adds query biophysics, within fixed eligibility cohorts. Long-range fractions use combined-class source bands, averaged over test proteins.

## 7. Target sensitivity and benchmarking

Tables S1–S2 need the saved residue predictions and the per-replicate ATLAS Protein Block assignments from the analysis archives:

```bash
python -m esmfluc.analysis.endpoint.analyze_neq_pb_reliability --help
```

Outputs include stricter endpoints, replicate-consensus performance and alignment/target-reconstruction audits.

The frozen mdCATH cohort has 989 domains; 985 have length-aligned Neq targets. Its filter removes PDB-code overlap, then MMseqs2 matches at 30% identity and 50% **query (ATLAS)** coverage (`--cov-mode 2`). `filter_provenance.json` and the cluster assignments record the cohort. Rebuilding it needs the mdCATH trajectories.

mdCATH Neq targets: for each domain in `data/mdcath/domains.txt`, Protein Blocks are assigned with PBxplore to every saved frame of each replica. Neq is computed per replica and averaged over the five replicas at each temperature. The first and last two residues have no Protein Block and get Neq = 1. Domains are then split by MMseqs2 cluster (seed 42); the 1,069-domain test set at 320 K is the cohort before ATLAS filtering. This downloads every mdCATH trajectory file:

```bash
pip install -e ".[mdcath]"
python -m esmfluc.datasets.compute_mdcath_neq --data_root data/mdcath
MMSEQS_BIN=/path/to/mmseqs python -m esmfluc.datasets.split_mdcath_clusters --data_root data/mdcath
python -m esmfluc.datasets.split_mdcath_by_temperature
```

PEGASUS and DynaMine scores come from their providers: the PEGASUS workflow records its download URL and Docker image, and the DynaMine client sends sequences to Bio2Byte. RMSF extraction downloads large files and asks for confirmation:

```bash
python -m esmfluc.benchmarks.benchmark_pegasus_mdcath --help
python -m esmfluc.datasets.fetch_dynamine_mdcath --help
python -m esmfluc.datasets.extract_mdcath_rmsf --help
python -m esmfluc.benchmarks.benchmark_unified_mdcath --help
python -m esmfluc.benchmarks.benchmark_atlas_mdcath_missing_metrics --help
```

Tables S7–S8 and Figure 2c,d use `unified_method_summary.csv`. Table S9 uses `results/benchmark_unified_mdcath/unified_paired_deltas.csv`, with paired-domain differences and cluster-bootstrap intervals. The metric-completion module computes the reported mdCATH AUROC tables. Table S10 repeats the Neq benchmark with `--exclude-ends 2` (without `--rmsf-csv`), written to `results/benchmark_unified_mdcath_ends_excluded/`.

## 8. Guanylate kinase

See `examples/gk_wt/FIGURE.md` for Figure 3, which combines the three-seed contribution matrix and bands with a ChimeraX view of PDB 1ZNX chain A.
