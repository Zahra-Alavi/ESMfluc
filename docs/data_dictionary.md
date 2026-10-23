# Data dictionary

The result index supplies exact files. CSVs have headers. Missing values are unavailable/undefined measurements, not zeros. Column names containing `flexible` or `rigid` refer to multi-PB- or single-PB-supporting evidence, not to every form of protein motion.

## Identifiers and coordinates

- `condition`: published model configuration; only six have task-specific attention contributions.
- `seed`: independent training seed 1, 2 or 3. `average` denotes a mean profile, never averaged model weights.
- `name` / `protein`: input FASTA or ATLAS identifier; `domain`: mdCATH identifier.
- `split`: fixed train, validation or test membership.
- `sequence_sha256`, `length`: checks of exact sequence identity and residue count.
- `union_group_id`: related proteins/domains grouped for split separation and group-level inference.
- `*_index_0based`: Python input-sequence coordinate; `*_residue_1based`: corresponding one-based sequence coordinate, **not necessarily PDB numbering**.
- `*_end_*_inclusive`: final included residue; the equivalent Python slice ends one position later.

## Contributions and bands

`C[i,j] = A[i,j] * s[j]` is signed evidence delivered from source/key `j` to query `i`. Positive supports multi-PB; negative supports single-PB. A row sum plus bias difference reconstructs the query margin. A column mean is source influence `I[j]`.

`stable_signed_bands.csv` records condition, split, protein, sign, primary apex, interval, amplitude and seed support. `primary_stable_band` requires at least two matching seeds; `strict_stable_band` requires all three. Protein summaries include zero-band proteins. Same-sign interval matching uses IoU at least 0.5. Mean contribution matrices are means of products; multiplying separately averaged attention and evidence is not equivalent.

## Local and structural effects

Read sign, metric and matching scheme together in the headline inference tables. Effects are peak minus matched control, with confidence limits, group counts and BH-adjusted results. RSA is a fraction, torsional change is in degrees, boundary distance is in residues, and C-alpha contacts are counts. Strain follows the producing MDStrainMapper definition. These are peak measurements, not band averages. An effect's direction and a band's contribution sign are separate quantities.

## Localization: Table S4

`per_apex_component_localization.csv` has one row per apex: `evidence_extremum`, `attention_maximum`, and category `evidence_only`, `attention_only`, `both` or `neither`. Stable summaries use `n_apices` as denominator. References use `n_positions`: other same-sign raw influence extrema excluding stable primary apices, or all same-sign interior positions. Evidence and attention counts each include `both`. Percentages are descriptive, not enrichment tests.

## Receivers: Table S5

Select test rows, stage `00_distance_q8` or `01_add_query_biophysics`, and the recorded `all_receivers` / `all_eligible_queries` cohort in `receiver_model_performance.csv`. The baseline uses source–query sequence-distance and query Q8 information; later stages add query biophysics and, separately, experimental structure/water features. Do not mix eligibility cohorts.

`long_range_receiver_summary.csv` supplies protein-averaged test fractions for combined-class source bands. The exact distance/strength thresholds are in the producing parameters. Sequence-distance scope does not establish physical interaction. High/low receiver labels refer to contribution strength.

## Predictive tables

- ATLAS `pooled_residue_performance_by_seed.csv`: pooled test AUROC, AUPRC and `neq_score_spearman`. The across-seed file reports mean and sample SD. Protein-macro files are separate.
- Endpoint tables: retain `cutoff`, target definition and score type. `auroc_ensemble` uses mean probabilities across seeds, unlike `auroc_seed_mean`.
- Consensus tables: retain `consensus_definition` and valid-replicate eligibility.
- `unified_per_protein_metrics.csv`: method/target/domain measurements; mismatched targets are excluded, not truncated.
- `unified_method_summary.csv`: select target (`neq` / `rmsf`) and scope (`method_available` / `shared_loaded_methods`). `macro_mean_spearman` is an equal-domain mean; intervals use cluster bootstrap, not training-seed SD.
- `unified_paired_deltas.csv`: paired ESMfluc-minus-external Spearman differences, domain/cluster counts and bootstrap intervals; the Wilcoxon p-value is explicitly unadjusted.
- `mdcath_neq_auroc_summary.csv`: distinguishes seed-specific, seed-averaged and unseeded external `aggregation` values.

## Provenance

Parameter and audit files record settings and input hashes. The model manifest in `provenance/model_release/` is for reference; restoring weights uses the manifest distributed with the Zenodo record. `mdstrainmapper.json` identifies the MDStrainMapper version used for the strain values.
