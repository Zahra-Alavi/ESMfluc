# Data dictionary

Exact files are listed in the result index. Missing values mean a measurement is unavailable, not zero. Column names with `flexible` or `rigid` refer to evidence supporting multi-PB or single-PB.

## Identifiers and coordinates

- `condition`: published model configuration; only six have task-specific attention contributions.
- `seed`: independent training seed 1, 2 or 3. `average` marks a mean over seeds.
- `name` / `protein`: input FASTA or ATLAS identifier; `domain`: mdCATH identifier.
- `split`: fixed train, validation or test membership.
- `sequence_sha256`, `length`: checks of exact sequence identity and residue count.
- `union_group_id`: related proteins/domains grouped for split separation and group-level inference.
- `*_index_0based`: Python input-sequence coordinate; `*_residue_1based`: corresponding one-based sequence coordinate, not necessarily PDB numbering.
- `*_end_*_inclusive`: final included residue; the equivalent Python slice ends one position later.

## Contributions and bands

`C[i,j] = A[i,j] * s[j]` is the signed contribution of source (key) `j` to query `i`. Positive supports multi-PB; negative supports single-PB. Row sums plus the bias difference give the query's logit margin. A column mean is source influence `I[j]`.

`stable_signed_bands.csv` records condition, split, protein, sign, primary apex, interval, amplitude and seed support. `primary_stable_band` requires at least two matching seeds; `strict_stable_band` requires all three. Protein summaries include zero-band proteins. Matching uses same-sign interval IoU of at least 0.5. Mean contribution matrices average the products, not attention and evidence separately.

## Local and structural effects

In the headline inference tables, read sign, metric and matching scheme together. Effects are peak minus matched control, with confidence limits, group counts and BH-adjusted results. RSA is a fraction, torsional change is in degrees, boundary distance is in residues, and C-alpha contacts are counts. Strain follows the MDStrainMapper definition. All values are measured at peaks.

## Localization: Table S4

`per_apex_component_localization.csv` has one row per apex: `evidence_extremum`, `attention_maximum`, and category `evidence_only`, `attention_only`, `both` or `neither`. Stable summaries use `n_apices` as denominator. References use `n_positions`: other same-sign raw influence extrema excluding stable primary apices, or all same-sign interior positions. Evidence and attention counts each include `both`.

## Receivers: Table S5

Select test rows, stage `00_distance_q8` or `01_add_query_biophysics`, and the recorded `all_receivers` / `all_eligible_queries` cohort in `receiver_model_performance.csv`. The baseline uses source–query sequence-distance and query Q8 information; later stages add query biophysics and, separately, experimental structure/water features. Compare stages within one cohort.

`long_range_receiver_summary.csv` supplies protein-averaged test fractions for combined-class source bands. The distance and strength thresholds are in the parameter files. High/low receiver labels refer to contribution strength.

## Predictive tables

- ATLAS `pooled_residue_performance_by_seed.csv`: pooled test AUROC, AUPRC and `neq_score_spearman`. The across-seed file reports mean and sample SD. Protein-macro files are separate.
- Endpoint tables: retain `cutoff`, target definition and score type. `auroc_ensemble` uses mean probabilities across seeds, unlike `auroc_seed_mean`.
- Consensus tables: retain `consensus_definition` and valid-replicate eligibility.
- `unified_per_protein_metrics.csv`: method/target/domain measurements; mismatched targets are excluded, not truncated.
- `unified_method_summary.csv`: select target (`neq` / `rmsf`) and scope (`method_available` / `shared_loaded_methods`). `macro_mean_spearman` is an equal-domain mean; intervals use the cluster bootstrap.
- `unified_paired_deltas.csv`: paired ESMfluc-minus-external Spearman differences, domain/cluster counts and bootstrap intervals; the Wilcoxon p-value is unadjusted.
- `mdcath_neq_auroc_summary.csv`: distinguishes seed-specific, seed-averaged and unseeded external `aggregation` values.

## Provenance

Parameter and audit files record settings and input hashes. The model manifest in `provenance/model_release/` is a reference copy; restoring weights uses the one on Zenodo. `mdstrainmapper.json` identifies the MDStrainMapper version used for the strain values.
