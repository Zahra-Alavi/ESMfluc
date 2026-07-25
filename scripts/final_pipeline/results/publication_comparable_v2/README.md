# Pipeline Summary:

```text
Fixed train / validation / test datasets
│
├── 1. Train comparable models
│   ├── ESM2 → BiLSTM → self-attention → classifier
│   │   ├── frozen
│   │   ├── top-4 fine-tuned
│   │   └── top-28 fine-tuned
│   ├── ESM3 → BiLSTM → self-attention → classifier
│   │   ├── frozen
│   │   ├── top-4 fine-tuned
│   │   └── top-28 fine-tuned
│   └── frozen ESM2 linear baseline
│
│   Each condition is trained with seeds 1, 2, and 3
│   → 7 conditions × 3 seeds = 21 runs
│
├── 2. Evaluate and extract model outputs
│   ├── checkpoint and test metrics
│   ├── hard residue predictions
│   ├── flexible-class probabilities
│   ├── complete class probabilities
│   ├── BiLSTM self-attention A
│   ├── exact signed contributions C
│   └── ESM2 backbone attention where applicable
│
├── 3. Test reproducibility across seeds
│   ├── predictive stability
│   ├── attention stability
│   ├── signed-contribution stability
│   └── ESM2 backbone-attention stability
│
├── 4. Extract signed contributions for all splits
│   ├── train
│   ├── validation
│   └── test
│
│   Produces, for every protein:
│   ├── attention routing A_ij
│   ├── intrinsic signed evidence s_j
│   ├── exact contribution C_ij = A_ij × s_j
│   ├── signed column influence I_j
│   └── attention-column mean B_j
│
└── 5. Signed-band analysis
    ├── Phase 1: detect bands and test seed reproducibility
    ├── Phase 2: annotate and analyze biophysical enrichment
    ├── Phase 3A: explain Q8-segment selection
    ├── Phase 3B: relate bands to experimental structure
    ├── Phase 3C: separate evidence from consultation
    ├── Phase 4: identify query residues receiving band contributions
    └── Phase 5: test sequence PWMs and motifs
```

## Implementation:

### 1. Train comparable models x 3 seeds

- esm2_frozen_linear
- esm2_frozen_bilstm_attn
- esm2_top4_bilstm_attn
- esm2_top28_bilstm_attn
- esm3_frozen_bilstm_attn
- esm3_top4_bilstm_attn
- esm3_top28_bilstm_attn

Each of the seven model conditions is trained with three random seeds, producing 21 runs in total.

### 2. Evaluate and extract model outputs

 After training, the pipeline evaluates each run and extracts its predictions, class probabilities, attention maps, and signed contributions.

### 3. Test reproducibility across seeds

`analyze_publication_seed_variance.py` then compares these outputs across seeds and model conditions. It writes the resulting summary tables to `results/publication_comparable_v2/analysis/`.
The files included on GitHub are the compact analysis summaries, not the model checkpoints or full attention matrices. They report per-seed and seed-averaged predictive performance, protein-level performance, within-condition reproducibility across seeds, and agreement between model conditions. The analysis parameters and a short run summary are also included for reproducibility.


| Model condition           | Accuracy        | Macro F1        | Flexible F1     | AUROC           | AUPRC           | Neq Spearman ρ  |
|---------------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| ESM2 BiLSTM-attn — frozen | 0.8065 ± 0.0024 | 0.7992 ± 0.0025 | 0.8374 ± 0.0026 | 0.8753 ± 0.0015 | 0.8880 ± 0.0008 | 0.6743 ± 0.0017 |
| ESM2 BiLSTM-attn — top-4  | 0.8036 ± 0.0010 | 0.7958 ± 0.0001 | 0.8356 ± 0.0026 | 0.8748 ± 0.0004 | 0.8875 ± 0.0002 | 0.6715 ± 0.0011 |
| ESM2 BiLSTM-attn — top-28 | 0.8087 ± 0.0005 | 0.8008 ± 0.0011 | 0.8404 ± 0.0004 | 0.8786 ± 0.0009 | 0.8908 ± 0.0007 | 0.6800 ± 0.0017 |
| ESM3 BiLSTM-attn — frozen | 0.7976 ± 0.0042 | 0.7889 ± 0.0042 | 0.8317 ± 0.0047 | 0.8651 ± 0.0020 | 0.8768 ± 0.0025 | 0.6544 ± 0.0041 |
| ESM3 BiLSTM-attn — top-4  | 0.8112 ± 0.0068 | 0.8044 ± 0.0068 | 0.8407 ± 0.0064 | 0.8822 ± 0.0049 | 0.8944 ± 0.0036 | 0.6849 ± 0.0077 |
| ESM3 BiLSTM-attn — top-28 | 0.8174 ± 0.0049 | 0.8117 ± 0.0052 | 0.8445 ± 0.0042 | 0.8895 ± 0.0052 | 0.9022 ± 0.0050 | 0.6976 ± 0.0081 |
| ESM2 linear — frozen      | 0.7920 ± 0.0008 | 0.7836 ± 0.0008 | 0.8262 ± 0.0008 | 0.8621 ± 0.0006 | 0.8756 ± 0.0010 | 0.6480 ± 0.0011 |

### 4. Extract signed contributions for all splits

### 5. Signed-band analysis


#### Phase 1: detect bands and test seed reproducibility

After signed contributions are extracted for the train, validation, and test sets, `add_seed_averaged_influence.py` averages each residue’s signed influence, \(I_j\), across seeds 1–3. `extract_signed_contribution_bands.py` uses this averaged profile to identify positive flexibility-supporting bands and negative rigidity-supporting bands.

`seed_averaged_signed_bands.csv.gz` contains the position, sequence, sign, width, magnitude, and persistence of every detected band. `seed_averaged_signed_band_protein_summary.csv` reports the number of positive and negative bands found for each protein. `signed_band_parameters.json` records how the bands were detected.

Seed reproducibility is evaluated separately by `analyze_signed_band_seed_reproducibility.py`. `consensus_reproducibility_summary.csv` summarizes bands supported across seeds, while `seed_pair_reproducibility_summary.csv` reports agreement between each pair of seeds.


#### Phase 2:

`annotate_signed_bands_with_netsurfp.py` adds ATLAS Neq and NetSurfP annotations, including secondary structure, solvent accessibility, disorder, interface, and torsion-related features. `analyze_signed_band_biophysical_enrichment.py` compares band apices with appropriate background residues and writes the enrichment summaries. `audit_signed_band_biophysical_pipeline.py` checks the complete analysis for missing data, coordinate errors, invalid matches, and inconsistent statistics.

The compact results are stored in `phase2_biophysical_results/`:

- `annotation_band_coverage_identifier_summary.csv` reports how well signed bands cover or identify annotated residue classes.
- `apex_circular_shift_enrichment_summary.csv` compares observed band-apex features with position-shifted null distributions.
- `paired_flex_vs_rigid_summary.csv` directly compares flexibility-supporting and rigidity-supporting bands.
- `within_q3_matched_enrichment_summary.csv` compares band apices with non-band residues matched by secondary structure and additional covariates.
- `within_q3_match_coverage_summary.csv` reports how many band apices received suitable matched controls.
- `within_q3_match_balance.csv` checks whether the matched cases and controls are comparable.
- `biophysical_enrichment_parameters.json` records the analysis settings and statistical procedures.
- `phase2_pipeline_audit.json` records the validation checks; all 42 checks passed.


#### Phase 3: Explaining signed-band selection and mechanism

Phase 3 examines why particular residue segments are selected as signed bands, whether those selections correspond to external structural properties, and whether their effects arise from intrinsic residue evidence or attention routing.

##### Phase 3A: Q8-segment selection

`analyze_signed_band_object_selection.py` treats complete contiguous Q8 segments as the analysis objects. Band-containing segments are compared with clean segments from the same protein with the same Q8 class.

- `matched_feature_effect_summary.csv` reports differences in flexibility, exposure, geometry, disorder, and sequence composition.
- `sequential_model_performance.csv` shows how each added feature group improves segment-selection performance.
- `sequential_model_coefficients.csv` contains the standardized coefficients from those models.
- `phase3a_parameters.json` records the segment definitions, matching rules, features, and model stages.

##### Phase 3B: External structure

`analyze_signed_band_external_structure.py` relates the Phase 3A segments to mapped experimental structures, contact networks, ECOD domain boundaries, and test-set strain data.

- `matched_external_effect_summary.csv` reports structural differences between selected and matched control segments.
- `sequential_external_model_performance.csv` tests whether external structural features add explanatory value.
- `mechanism_class_external_associations.csv` connects the Phase 3C mechanism classes with external structural features.
- `external_feature_coverage_summary.csv` reports structural and strain-data availability.
- `structure_mapping_audit.csv` records mapping quality and acceptance for each protein.
- `parameters.json` records the structural thresholds, feature groups, and model settings.

##### Phase 3C: Evidence versus attention routing

`analyze_signed_band_model_mechanism.py` uses the identity \(I_j=s_jB_j\) to determine whether each band is driven mainly by intrinsic signed evidence, attention consultation, or a mixture of both.

- `mechanism_effect_summary.csv` reports matched changes in evidence, attention breadth, and signed influence.
- `mechanism_class_summary.csv` reports the number and fraction of evidence-dominated, attention-dominated, and mixed bands.
- `phase3c_parameters.json` records the decomposition and classification rules.

#### Phase 4: Query residues receiving band contributions

`analyze_signed_band_query_receivers.py` performs the receiver analysis separately for each model condition and seed, averages the results across seeds, compares high- and low-receiving queries, and tests which query features help explain receiver selection.

The results in `nonstructural_receiver_analysis/` contain the completed original analysis. This version uses sequence distance, Q8 class, Neq, RSA, disorder, torsion change, amino-acid class, sequence position, and band membership. It does not include experimental-structure, contact-network, or crystallographic-water features.

- `receiver_feature_summary.csv` reports which query features differ between high- and low-receiving residues.
- `long_range_receiver_summary.csv` summarizes how much receiver activity and contribution mass occurs at long sequence distances.
- `receiver_model_performance.csv` reports the performance of models trained to identify high-receiving query residues.
- `receiver_feature_ablation_performance.csv` measures the effect of adding or removing individual query features.
- `receiver_feature_effects_by_protein.csv.gz` contains the protein-level effects underlying the summary statistics.
- `parameters_by_condition.json` records the receiver definitions, thresholds, matching rules, features, and seed handling.
- `extraction_audit_by_condition.json` records extraction coverage and contribution-reconstruction accuracy.

The upgraded structural receiver analysis will be added separately after all six model conditions and its final audit are complete.

#### Phase 5: Sequence PWMs and motifs

##### Apex-centered PWMs and motifs

`analyze_signed_band_apex_pwm.py` examines amino-acid windows centered on each band apex. Band windows are compared with same-protein control windows matched by Q8 secondary-structure class.

- `pwm_amino_acid_frequencies.csv.gz` contains position-specific amino-acid frequencies and enrichment around flexibility-supporting and rigidity-supporting apices.
- `pwm_coverage_summary.csv` reports how many bands had suitable matched background windows.
- `kmer_cross_split_replication.csv.gz` contains exact amino-acid k-mers discovered in training and their validation/test results.
- `kmer_replication_summary.csv` summarizes how many exact k-mers replicate across splits.
- `reduced_motif_cross_split_replication.csv.gz` contains motifs built from biochemical amino-acid classes and their held-out results.
- `reduced_motif_replication_summary.csv` summarizes reduced-motif replication.
- `reduced_alphabet_mapping.csv` defines the biochemical amino-acid classes.
- `parameters_by_condition.json` records the PWM, matching, discovery, and replication settings for each model condition.

##### Complete-segment sequence motifs

`analyze_signed_band_sequence_motifs.py` compares band-containing Q8 segments with unselected segments from the same protein and Q8 class. Motifs are selected using training proteins and locked before validation and test evaluation.

- `locked_motif_definitions_by_condition.json` contains the motifs selected from the training split.
- `motif_validation_test_results.csv` reports held-out validation and test performance for each locked motif.
- `motif_model_family_consistency.csv` tests whether motif direction is consistent across ESM2 and ESM3 models.
- `motif_incremental_model_performance.csv` tests whether sequence motifs improve segment-selection models beyond the biophysical features.
- `parameters_by_condition.json` records the matching, discovery, filtering, and evaluation settings.
