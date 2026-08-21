# ESMfluc signed-contribution / I_j band analysis

## REPOSITORY ROOT

```text
/home/zahralab/Desktop/ESMfluc/scripts/final_pipeline
```

All relative paths below are relative to this directory.

## SCIENTIFIC OBJECTIVE

Identify protein residues or regions acting as influential attention keys in
the BiLSTM-attention models, separate flexibility-supporting (+) from
rigidity-supporting (-) keys, determine whether their locations are
reproducible and biologically nonrandom, and explain:

1. Their local biophysical environments.
2. Why one plausible structural object is selected as a band while another
   similar object in the same protein is not.
3. Whether band strength comes from intrinsic signed evidence, broad attention
   consultation, or both.

Important terminology:

- “Positive” or “+” means flexibility-supporting.
- “Negative” or “-” means rigidity-supporting.
- These labels describe the model’s signed logit contribution, not necessarily
  the experimentally observed state of the band residue itself.

## CORE DEFINITIONS

For query residue i and key residue j:

    C_ij = A_ij * s_j

    I_j = (1/L) sum_i C_ij
        = s_j * [(1/L) sum_i A_ij]
        = s_j * B_j

where:

- A_ij:
  Attention from query residue i to key residue j.

- s_j:
  Intrinsic signed logit evidence at key j.

- C_ij:
  Exact signed contribution from key j to query i.

- I_j:
  Signed column influence of key j across all query residues.

- B_j:
  Attention-column mean:

      B_j = (1/L) sum_i A_ij

  This measures how broadly key j is consulted across the protein.

Interpretation:

- Positive I_j: flexibility-supporting key.
- Negative I_j: rigidity-supporting key.
- The sign of I_j is determined by s_j because B_j is nonnegative.
- B_j modulates the magnitude of the influence.

## ENSEMBLE RULE

Final bands were detected from the three-seed arithmetic average:

    mean_seed(I_j)
      = [I_j(seed1) + I_j(seed2) + I_j(seed3)] / 3

Per-seed bands were analyzed first to verify reproducibility. The final
biophysical analyses use bands detected from mean_seed(I_j).

## DATASETS AND MODELS

Dataset splits:

- train: 967 proteins, 223,573 residues
- validation: 208 proteins, 47,875 residues
- test: 208 proteins, 47,751 residues
- total unique proteins: 1,383
- total unique residue annotations: 319,199

Six model conditions:

- esm2_frozen_bilstm_attn
- esm2_top4_bilstm_attn
- esm2_top28_bilstm_attn
- esm3_frozen_bilstm_attn
- esm3_top4_bilstm_attn
- esm3_top28_bilstm_attn

Three seeds per condition:

- 18 model runs
- 54 condition/seed/split contribution files
- 24,894 protein-level inferences
- approximately 36.80 GB compressed

Maximum audited reconstruction error for:

    I_j = s_j * B_j

was 1.91e-6.

## PRIMARY DATA LOCATIONS

Fixed grouped datasets:

```text
data_splits/atlas_grouped_v1/
  train_grouped_v1.csv
  validation_grouped_v1.csv
  test_grouped_v1.csv
  train_grouped_v1.fasta
  validation_grouped_v1.fasta
  test_grouped_v1.fasta
```

NetSurfP predictions for these exact sequences are:

```text
data_splits/atlas_grouped_v1/
  train_grouped_v1_netsurfp3.json
  validation_grouped_v1_netsurfp3.json
  test_grouped_v1_netsurfp3.json
```

Split membership, group IDs and leakage checks are recorded in:

```text
data_splits/atlas_grouped_v1/
  split_manifest_grouped_v1.csv
  group_manifest_grouped_v1.csv
  split_summary_grouped_v1.json
  excluded_entries.csv
```

Training outputs for all seven model conditions and three seeds are under:

```text
results/publication_comparable_v2/runs/<condition>/seed_<seed>/
```

There are 21 complete run directories. Each contains the checkpoint,
training log, run arguments, prediction metrics and test inference output. The
frozen ESM2 linear baseline is included here, but it has no attention or exact
contribution matrices.

Exact signed contributions are available for the six BiLSTM-attention
conditions and three seeds. The manifest for all 54 condition/seed/split files
is:

```text
results/publication_comparable_v2/
  all_split_signed_contributions_manifest.tsv
```

The corresponding extraction audit is:

```text
results/publication_comparable_v2/
  all_split_signed_contributions_audit.json
```

The files themselves are stored under:

```text
results/publication_comparable_v2/runs/<condition>/seed_<seed>/
  all_split_signed_contributions/
    train_signed_contributions.json.gz
    validation_signed_contributions.json.gz
    test_signed_contributions.json.gz
```

Every protein record contains:

- `contribution_matrix`: exact `C_ij`, shape `L x L`;
- `intrinsic_signed_evidence`: `s_j`, shape `L`;
- `signed_column_influence`: seed-specific `I_j`, shape `L`;
- `attention_matrix`: `A_ij`, shape `L x L`;
- `attention_column_mean`: `B_j`, shape `L`;
- `seed_averaged_signed_column_influence`: the three-seed mean of `I_j`,
  shape `L`.

The same three-seed mean field was inserted into each seed's contribution
file. The following manifest points to the seed-1 copy as the canonical file
for reading the averaged profile; the stored mean still comes from seeds 1,
2 and 3:

```text
results/publication_comparable_v2/
  seed_averaged_signed_contributions_manifest.tsv
```

The seed-average insertion audit is:

```text
results/publication_comparable_v2/
  seed_averaged_influence_audit.tsv
```

The current final stable-band catalog is:

```text
results/publication_comparable_v2/
  analysis_phase1_upgraded_raw_mad2/
    reproducibility_interval_iou05_null_fixed/
      stable_signed_bands.csv
```

This file contains train, validation and test bands. Test-only confirmatory
null results are under `reproducibility_interval_iou05_null_fixed_test/`.

Current primary downstream result directories are:

```text
results/publication_comparable_v2/
  analysis_neq_pb_reliability/
  analysis_phase2_primary_no_position_test/
  analysis_apex_structure_primary_no_position_test/
  analysis_phase3c_interval_iou05_stable_all_splits/
  analysis_phase4_interval_iou05_stable_query_receivers_upgraded/

results/benchmark/
```

The older similarly named analysis directories are retained for provenance,
but these are the directories to use for the current manuscript.

## DATA-GENERATION SCRIPTS

```text
Attention/extract_all_split_signed_contributions.py
```

```text
Attention/audit_all_split_signed_contributions.py
```

```text
signed_band_analysis/add_seed_averaged_influence.py
```

The extraction runner used both GPUs:

```text
run_all_split_signed_contributions_v2.sh
```

## PHASE 1: BAND DETECTION AND SEED REPRODUCIBILITY

Band detector:

```text
signed_band_analysis/extract_signed_contribution_bands.py
```

### 1A. Band-calling algorithm

The current publication detector uses the raw residue-level influence profile
without smoothing:

```text
detector_version: phase1_raw_amplitude_half_intensity_v1
locked_parameter_set_id: phase1_raw_amp_R2_half_intensity_merge_c5d7c802e9
```

For each protein profile, the robust scale is:

```text
sigma_MAD = 1.4826 * median(|I_j - median(I)|)
R_p = |I_p| / sigma_MAD
```

The median and MAD are calculated over the detector's eligible profile slice.
Because both numerator and denominator scale together, the detector is exactly
invariant to multiplying a complete profile by a positive constant. It detects
relative profile shape, not absolute influence magnitude.

Algorithm:

1. Find positive local maxima and negative local minima in the raw `I_j`
   profile. The first and last residues are not eligible as apices.
2. Keep an apex when its absolute influence is at least twice the profile's
   robust MAD scale: `R_p >= 2`.
3. Define its band as the contiguous, same-sign residues around the apex for
   which `|I_j| >= 0.5 * |I_p|`.
4. Merge overlapping bands of the same sign and retain the strongest apex as
   the primary apex.

Each final band contains its apex, contains residues of only one sign and does
not overlap another final band. The per-seed and mean-profile interval audits
passed with no overlapping or multiply assigned residues.

The output also records band strength. The main measures are absolute apex
influence, standardized apex magnitude `R_p`, integrated absolute influence
across the band, and their within-protein ranks. These values can be used to
distinguish stronger from weaker detected bands in downstream analyses.


### 1B. Per-seed bands

Outputs:

```text
results/publication_comparable_v2/analysis_phase1_upgraded_raw_mad2/per_seed/
  signed_bands.csv
  signed_band_protein_summary.csv
  signed_band_parameters.json
  band_interval_audit.json
```
Each row includes identifying columns such as:
- condition: which of the six models
- seed: 1, 2, or 3
- split: train, validation, or test
- protein: the protein identifier


### 1C. Seed reproducibility

Script:

```text
signed_band_analysis/analyze_signed_band_seed_reproducibility.py
```

Outputs:

```text
results/publication_comparable_v2/
  analysis_phase1_upgraded_raw_mad2/
    reproducibility_interval_iou05_null_fixed/
      all_mean_bands_with_stability.csv.gz
      seed_consensus_bands.csv
      seed_pair_reproducibility_by_protein.csv
      stable_signed_bands.csv

    reproducibility_interval_iou05_null_fixed_test/
      all_mean_bands_with_stability.csv.gz
      seed_pair_reproducibility_by_protein.csv
      seed_pair_reproducibility_summary.csv
      seed_consensus_bands.csv
      consensus_reproducibility_summary.csv
      seed_pair_block_shift_null.csv
      consensus_block_shift_null.csv
      seed_reproducibility_parameters.json
      stable_signed_bands.csv
```

Seed matching:

- Bands are matched only within the same model condition, split, protein and
  sign.
- Matching is one-to-one and requires interval IoU of at least 0.5.
- A mean-profile band is stable when it matches a band in at least two of the
  three seed profiles. Support from all three seeds is recorded separately.
- Band strength, rank and prominence do not affect seed matching.

`seed_consensus_bands.csv` is a diagnostic summary of where the three individual seed runs detected similar bands.

The circular-shift null preserves each protein's band counts, signs, widths and
relative spacing while shifting their locations. It tests whether the observed
agreement across seeds is greater than expected from patterns with the same
basic organization but unrelated residue positions. The stable-band catalog
does not depend on the null result.

Inferential test-set results across 36 condition/sign/seed-pair cells:

- pairwise micro-Jaccard: 0.427–0.880; median 0.679;
- pairwise micro-F1: 0.599–0.936; median 0.809;
- mean matched-apex separation: 0.19–1.28 residues; median 0.67;
- mean matched interval IoU: 0.668–0.935; median 0.802;
- pairwise Jaccard null z-scores: 75.5–153.6;
- consensus-count null z-scores: 43.0–61.5;
- all pairwise and consensus BH-adjusted empirical q-values were `0.000999`
  with 1,000 block shifts.

The raw-profile band locations and intervals are therefore strongly
reproducible across seeds and exceed the circular-shift expectation.

### 1D. Final band catalog

The same detector was applied to the arithmetic mean of the three seed
profiles.

Outputs:

```text
results/publication_comparable_v2/
  analysis_phase1_upgraded_raw_mad2/
    mean/
      signed_bands.csv
      signed_band_protein_summary.csv
      signed_band_parameters.json
      band_interval_audit.json

    reproducibility_interval_iou05_null_fixed/
      all_mean_bands_with_stability.csv.gz
      stable_signed_bands.csv
```

The detector found 77,085 bands in the seed-averaged profiles before the seed
stability filter. Of these, 72,465 were supported by at least two seeds and form
the final catalog.

Final counts:

- 72,465 total stable bands (across 6 models)
- 44,712 positive bands
- 27,753 negative bands
- 3.784 total bands per 100 residues
- positive density: 2.335 per 100 residues
- negative density: 1.449 per 100 residues
- train: 50,395 bands
- validation: 11,031 bands
- test: 11,039 bands
- 55,170 bands supported by all three seeds
- 17,295 bands supported by two seeds

The densities use the same denominator as the counts: residue positions summed
over the six separate model profiles. They are not densities of unique regions
after combining the six models.

Widths:

- positive mean width: 4.47 aa
- positive median width: 4 aa
- negative mean width: 6.40 aa
- negative median width: 5 aa

Apex distances between consecutive bands in the complete ordered band list:

- +/+ mean 26.79, median 22 residues
- -/- mean 29.93, median 25 residues
- opposite-sign mean 21.51, median 16 residues

Same-sign distance while ignoring intervening opposite-sign bands:

- + to next +: mean 38.38, median 29 residues
- - to next -: mean 48.71, median 38 residues


The primary downstream catalog is:

```text
results/publication_comparable_v2/
  analysis_phase1_upgraded_raw_mad2/
    reproducibility_interval_iou05_null_fixed/stable_signed_bands.csv
```

Test-only confirmatory analyses may use the identical test subset under
`reproducibility_interval_iou05_null_fixed_test/`.

### 1E. Evidence-only and shifted-attention controls

The observed influence profile combines two learned quantities:

\[
I_j^{\mathrm{observed}} = s_j B_j,
\qquad
B_j = \frac{1}{L}\sum_i A_{ij}.
\]

Here, \(s_j\) is the signed evidence stored at residue \(j\), \(A_{ij}\) is the
attention from query residue \(i\) to key residue \(j\), \(B_j\) is the average
attention received by residue \(j\), and \(L\) is the protein length. The
observed profile can therefore highlight a residue because it carries strong
evidence, because attention routes many queries toward it, or because of both.

Two post hoc controls separate different explanations for band localization.
The uniform-attention control asks whether the learned evidence alone is
sufficient. Under uniform attention, every key receives \(1/L\):

\[
I_j^{\mathrm{uniform}} = \frac{s_j}{L}.
\]

The shifted-attention control retains the learned shape and variation of
\(B_j\), but breaks its residue-wise alignment with \(s_j\):

\[
I_j^{\mathrm{shifted}} = s_j\,
\operatorname{circularShift}(B_j).
\]

Each protein and model condition receives a reproducible nonzero circular
shift. The same shift is used for all three training seeds so that the control
does not manufacture seed instability. This is a fairer control for the fact
that multiplying by any nonuniform positive profile can create or amplify
local extrema.

Both controls use saved quantities from the original model. Neither is a
retrained model or a causal attention ablation, because \(s_j\) and \(B_j\)
were produced together by the trained network.

The attention amplification factor is:

\[
G_j = L B_j,
\qquad
I_j^{\mathrm{observed}} = I_j^{\mathrm{uniform}}G_j.
\]

Thus, \(G_j>1\) means that attention amplifies the evidence at residue \(j\)
relative to uniform routing, while \(G_j<1\) means that it suppresses it.

Implementation:

```text
signed_band_analysis/build_uniform_attention_control_profiles.py
signed_band_analysis/compare_observed_uniform_bands.py
```

The same detection and comparison procedure was used for both controls:

1. Build the control profile for every seed, model condition and protein. The
   uniform control was built for train, validation and test; the shifted-
   attention control was built for test only.
2. Average the three control profiles for each model and protein.
3. Apply the same locked Phase 1 detector used for the observed profiles:
   raw residue-level profiles, \(R_p\geq2\), and same-sign half-height band
   boundaries.
4. Apply the same seed-stability rule: interval IoU at least 0.5 and support in
   at least two of three seeds.
5. Compare observed and control profiles, band masks, apex locations, catalog
   sizes and seed support within the same model, split, protein and sign.

No test-set result was used to choose the detector or matching parameters. The
test split is the primary set for scientific interpretation. The shifted-
attention stability run used one null shift only, so its saved null statistics
are noninferential; the stable-band catalog itself does not depend on the number
of null shifts.

Outputs:

```text
results/publication_comparable_v2/
  analysis_uniform_attention_control_upgraded_raw_mad2/
    profiles/
    seed_averaged_profiles/
    profile_comparison/
    bands/per_seed_uniform/
    bands/mean_uniform/
    stability_uniform_interval_iou05_null_fixed/
    final_observed_uniform_comparison/

  analysis_shifted_attention_control_raw_mad2/
    profiles/
    seed_averaged_profiles/
    profile_comparison/
    bands/per_seed_shifted/
    bands/mean_shifted/
    stability_shifted_interval_iou05/
    final_observed_shifted_comparison/
```

Across all six models and all three data splits, the uniform control produced
13,518 seed-averaged candidate bands, of which 11,605 were seed-stable. For
comparison, the observed profiles produced 77,085 candidate bands, of which
72,465 were seed-stable. These are technical catalog counts summed across
models and splits, not counts of unique biological regions.

The complete-catalog comparison showed a strong asymmetry:

- 94.1% of uniform candidate bands overlapped an observed candidate band, but
  only 16.5% of observed candidate bands overlapped a uniform band;
- 94.1% of uniform stable bands overlapped an observed stable band, but only
  15.1% of observed stable bands overlapped a uniform stable band.

The complete residue profiles were nevertheless strongly correlated on the
test set. Across models, mean observed-versus-uniform Pearson correlations
were 0.912–0.949 for negative profiles and 0.868–0.918 for positive profiles;
the corresponding Spearman correlations were 0.995–1.000 and 0.987–0.999.
Thus, attention usually preserves the broad ranking already present in
\(s_j\), while changing which local peaks cross the band threshold and how
widely those peaks extend.

Test-set residue-level band overlap across the six models was:

| Catalog | Sign | Jaccard range | Observed residues covered by uniform | Uniform residues covered by observed |
| --- | ---: | ---: | ---: | ---: |
| Seed-averaged candidates | Negative | 0.204–0.335 | 0.286–0.531 | 0.596–0.655 |
| Seed-averaged candidates | Positive | 0.037–0.112 | 0.049–0.142 | 0.542–0.659 |
| Seed-stable bands | Negative | 0.182–0.300 | 0.252–0.479 | 0.586–0.675 |
| Seed-stable bands | Positive | 0.034–0.101 | 0.049–0.131 | 0.541–0.638 |

On the test set, the observed profiles also produced more candidates per
protein than the uniform profiles in every model. The ranges across models
were 3.28–3.75 observed versus 0.76–2.00 uniform for negative bands, and
5.43–6.08 observed versus 0.27–0.65 uniform for positive bands. Uniform bands
also had lower three-seed support overall, especially for positive bands.

The uniform bands therefore form a relatively small subset of the observed
bands. This shows that \(s_j\) contains much of the broad residue ranking, but
does not by itself reproduce most detected bands.

The shifted-attention control gives a more conservative comparison. On the
test set, the observed profiles produced 11,740 seed-averaged candidate bands
and 11,039 seed-stable bands. The shifted profiles produced 5,629 candidates
and 3,630 stable bands. These totals are summed across the six model
conditions, not counts of unique biological regions.

Test-set residue-level overlap with the shifted control was:

| Catalog | Sign | Jaccard range | Observed residues covered by shifted control | Shifted-control residues covered by observed |
|:---|---:|---:|---:|---:|
| Seed-averaged candidates | Negative | 0.357–0.402 | 0.506–0.620 | 0.542–0.650 |
| Seed-averaged candidates | Positive | 0.156–0.222 | 0.184–0.265 | 0.640–0.732 |
| Seed-stable bands | Negative | 0.265–0.333 | 0.369–0.440 | 0.569–0.658 |
| Seed-stable bands | Positive | 0.095–0.163 | 0.108–0.197 | 0.645–0.743 |

A separate diagnostic calculation on the current ESM3 top-28 test catalog
clarifies the mechanism. Among 1,868 seed-stable observed apices, 91.4% were
also a sign-appropriate raw local extremum of \(s_j\), 97.3% were a raw local
maximum of \(B_j\), 89.3% were both and 0.6% were neither. Yet applying the
locked detector to \(s_j/L\) produced only 346 seed-averaged candidates, 17.1%
of the 2,029 observed candidates for this condition. Thus, the component
profiles already contain nearly all candidate apex locations, while their
learned alignment determines which candidates become strong and stable enough
to pass the \(R_p\geq2\) detector. This diagnostic has not yet been exported as
a standalone result table by the pipeline.

Thus, much of the shifted-control catalog is contained within the observed
catalog, while many observed locations are lost when the alignment between
\(s_j\) and \(B_j\) is broken. Learned alignment contributes localization,
especially for positive bands, but the effect is smaller than the uniform
control alone suggested. The defensible interpretation is that attention
selects and amplifies evidence-bearing candidate locations; these controls do
not show that attention creates a biological communication network.

These controls answer a different question from Phase 3C. Phase 3C starts from
the observed bands and asks whether their strength is associated with
attention routing, signed evidence, or both. Phase 1E instead rebuilds complete
counterfactual profiles and asks which locations remain detectable.

## PHASE 2: BIOPHYSICAL ENRICHMENT, NONRANDOMNESS AND IDENTIFIER ANALYSIS

Phase 2 describes the biophysical environments of the seed-stable
contribution bands. Band and residue annotations are prepared by:

```text
signed_band_analysis/annotate_signed_bands_with_netsurfp.py
```
The statistical analyses are performed by:

```text
signed_band_analysis/analyze_signed_band_biophysical_enrichment.py
```

The publication analysis uses the stable bands detected from seed-averaged
profiles and supported by at least two of the three seeds. Its main statistical
conclusions are evaluated on the held-out test proteins.

The phase has five main questions:

2A. Where are positive and negative band apices located biophysically?

2B. Are those locations nonrandom relative to a protein-preserving positional
    null?

2C. What fraction of biological annotations is captured by bands? This is coverage or recall. (band detects annotation)

2D. After comparing apices with same-protein, same-Q3 non-band residues, what
    properties still distinguish band apices?

2E. On the test set, are positive and negative band apices associated with different MD-derived strain environments?

### 2A. Residue and band annotation

Inputs:

- final seed-averaged bands
- grouped-v1 experimental Neq
- NetSurfP Q3
- NetSurfP Q8
- RSA and ASA
- disorder
- interface
- phi and psi

Annotation outputs:

```text
results/publication_comparable_v2/
  analysis_phase2_interval_iou05_test/annotations/
    residue_biophysical_annotations.csv.gz
    signed_bands_biophysical_annotations.csv.gz
    annotation_audit.json
```

The equivalent all-split residue annotation table used by later phases is
under `analysis_phase2_upgraded_raw_mad2/annotations/`.

Derived annotations include:

- Q3 and Q8 segment boundaries
- distance to Q3 and Q8 boundaries
- whether a residue is within two residues of a boundary
- Q8 C/T/S loop-turn-bend label
- Neq peaks
- distance to the nearest Neq peak
- normalized sequence position
- torsional change from the preceding residue
- structured-linker/loop status

Torsional change is:

    sqrt[
      circular_delta(phi_j, phi_(j-1))^2
      +
      circular_delta(psi_j, psi_(j-1))^2
    ]

It measures how much the backbone torsion orientation changes from the
preceding residue to the current residue. Circular angular differences are
used so that, for example, -179 degrees and +179 degrees are treated as being
two degrees apart rather than 358 degrees apart.

“Structured linker/loop” is defined as:

- a contiguous Q8 C/T/S run
- length 2–20 residues
- with Q3 H or E found within three residues on both sides

It is therefore not just the Q8 label at the apex. It asks whether the apex is
part of a short C/T/S segment connecting structured regions.

### 2B. Raw apex localization and circular-shift nonrandomness

This analysis uses the 11,039 final stable bands from the held-out test set.
For each protein, model condition and sign, the script measures the annotations
at the observed band apices. It then circularly shifts the complete same-sign
apex pattern within the eligible protein interval and measures the annotations
at the shifted positions.

The null preserves:

- protein identity
- model condition
- sign
- number of bands
- relative band spacing
- clustering pattern

It changes only the absolute positions of the apex pattern.

Circular shifting is used only as a positional control. It does not change the
observed bands or determine which bands enter the stable catalog.

The analysis uses:

- 1,000 circular block shifts
- equal protein weighting
- protein-level observed means
- empirical upper, lower and two-sided p-values
- Benjamini-Hochberg correction

Test-set results across the six model conditions:

| Property at apex | Positive observed | Positive null | Negative observed | Negative null |
|:---|---:|---:|---:|---:|
| Neq | 2.061–2.151 | 1.368–1.370 | 1.00002–1.00046 | 1.322–1.331 |
| Q3 coil | 94.91–95.93% | 43.69–43.95% | 0–0.37% | 41.26–42.15% |
| Q8 C/T/S | 93.08–94.47% | 41.46–41.73% | 0–0.37% | 39.31–40.19% |
| Structured linker/loop | 63.83–73.75% | 29.07–29.56% | 0–0.30% | 28.69–29.40% |
| RSA | 0.537–0.555 | 0.336–0.339 | 0.178–0.270 | 0.329–0.331 |
| RSA ≥ 0.25 | 91.32–92.29% | 58.51–58.93% | 30.05–49.22% | 56.85–57.32% |
| Distance to Neq peak (aa) | 1.63–2.39 | 5.08–5.27 | 8.62–9.53 | 4.96–5.31 |
| Torsional change (degrees) | 90.66–96.47 | 47.48–47.85 | 2.60–4.58 | 46.04–47.03 |

The positive and negative null ranges are reported separately because the two
signs have different observed apex patterns and are shifted independently.
For every property in the table, all six model conditions differed from their
sign-matched shifted controls after multiple-testing correction. The absolute
null z-scores were 13.0–30.3 for positive apices and 3.24–19.6 for negative
apices; all two-sided BH-adjusted p-values were at most 0.00221.

Interpretation:

- Positive apices preferentially occupy flexible, exposed, loop-like,
  torsionally changing environments near Neq peaks.
- Negative apices preferentially occupy rigid, structured, less exposed
  environments with small local torsional changes.
- The apex locations are strongly nonrandom.
- These raw differences do not establish a specialized control mechanism
  because they partly reflect the flexible/rigid state associated with the
  signed contribution.

Accordingly, Phase 2B is treated as a model-localization and accuracy
diagnostic. The strict matched analysis in Phase 2D, especially external MD
strain, carries the main biological interpretation.

The script also performs paired positive-versus-negative protein-level
contrasts. The current outputs are stored in:

```text
results/publication_comparable_v2/analysis_phase2_primary_no_position_test/
  enrichment_with_test_strain/
    apex_metrics_by_protein.csv.gz
    apex_circular_shift_null.csv.gz
    apex_circular_shift_enrichment_summary.csv
    paired_flex_vs_rigid_summary.csv
```

### 2C. Inverse coverage and identifier analysis

Phase 2B asks:

    P(annotation | band)

For example:

    What percentage of positive apices are Q8 C/T/S?

Phase 2C reverses the question and estimates:

    P(detected by a band | annotation)

For example:

    What percentage of all Q8 C/T/S residues or segments are identified by a
    positive apex or band interval?

This distinction separates precision from coverage. A high precision means
that a detected apex is usually in the annotation of interest. High coverage
would mean that bands identify most occurrences of that annotation. Proteins
with zero bands are retained, so coverage is not inflated by analyzing only
proteins in which the model detected a band.

Detection methods include:

- exact apex
- within one residue of an apex
- within two residues
- within five residues
- overlap with a full band interval
- segment contains an apex
- segment overlaps a band interval

Outputs:

```text
results/publication_comparable_v2/
  analysis_phase2_primary_no_position_test/
    enrichment_with_test_strain/
      annotation_band_coverage_by_protein.csv.gz
      annotation_band_coverage_identifier_summary.csv
```

The following are test-set macro estimates: each protein receives equal weight,
and each range covers the six model conditions.

Positive-band results for Q8 C/T/S:

| Measurement | Result across models |
|:---|---:|
| Positive apices that are Q8 C/T/S | 93.08–94.47% |
| All Q8 C/T/S residues detected by an exact positive apex | 5.48–6.34% |
| Residues inside positive-band intervals that are Q8 C/T/S | 88.70–90.14% |
| All Q8 C/T/S residues covered by positive-band intervals | 21.10–28.05% |
| Q8 C/T/S segments containing a positive apex | 28.83–33.09% |
| Q8 C/T/S segments overlapping a positive-band interval | 31.55–35.69% |

Positive-band results for Neq peaks:

| Measurement | Result across models |
|:---|---:|
| Positive apices that are Neq peaks | 18.34–25.32% |
| All Neq peaks detected by an exact positive apex | 6.35–9.48% |
| All Neq peaks within two residues of a positive apex | 27.25–31.05% |
| All Neq peaks covered by positive-band intervals | 25.40–30.23% |

Conclusion:

- A positive apex is usually loop-like.
- Most loop-like residues and segments do not contain an apex.
- Positive bands are not general Q8-loop or Neq-peak detectors.
- The updated sign-constrained bands are compact. Compared with the legacy
  wider bands, they have higher Q8 C/T/S interval precision but cover a smaller
  fraction of all Q8 C/T/S residues and Neq peaks.
- This asymmetry motivated Phase 3A: why is one plausible Q8 segment selected
  while another same-Q8 segment in the same protein is not?

### 2D. Same-protein, within-Q3 matched analysis

Four matching schemes are implemented:

1. q3_only
2. q3_neq
3. q3_neq_rsa
4. q3_neq_rsa_position

All schemes require:

- same protein
- same model condition
- identical Q3 label
- control outside every positive and negative band interval
- case and control outside the first and last two sequence positions

The q3_only design uses:

- case: band apex
- controls: every eligible same-Q3 residue in the same protein
- controls are not matched on Neq, RSA or position
- Neq, RSA and position remain outcomes that can potentially explain band
  selection

The stricter schemes add:

- Neq caliper: 0.25
- RSA caliper: 0.15
- normalized-position caliper: 0.25
- up to five nearest eligible controls per apex

Sequence position is not used in the primary comparison. The first and last
two residues are excluded directly because these positions lack assigned PB
states. The position-matched scheme is retained only as a sensitivity
analysis.

Control reuse is allowed across different apex match sets but not within one
set. Candidate controls are ordered first by matching distance. Exact ties are
resolved by a reproducible SHA-256-derived random value keyed by the analysis
seed, condition, protein, band and candidate residue. Residue index is not used
as the tie-break, avoiding systematic selection toward the N terminus.

Neq, RSA and normalized position remain in the balance tables for every
scheme. They are omitted from the inferential outcome table whenever that
variable was used for matching. A residual difference inside a matching
caliper is therefore not reported as a biological finding.

The q3_only results are stored compactly as one case row containing the number
and means of all eligible controls. More than one million redundant
individual-control rows are not stored.

Q3-only coverage:

- total test-set apices across the six models: 11,039
- matched apices: 11,039
- match rate: 100%
- implied control assignments: 1,112,177
- mean controls per matched apex: 100.75
- every control comes from the same protein

Coverage under the stricter schemes:

| Matching scheme | Matched apices | Match rate | Mean controls per matched apex |
|:---|---:|---:|---:|
| Q3 only | 11,039 | 100% | 100.75 |
| Q3 + Neq | 10,537 | 95.45% | 4.61 |
| Q3 + Neq + RSA | 9,889 | 89.58% | 4.22 |
| Q3 + Neq + RSA + position | 8,828 | 79.97% | 3.83 |

The 11,039 total is an aggregate over six model conditions. The stricter schemes lose cases when no non-band
residue satisfies all required calipers.

Effects are calculated within protein. For the primary analysis, protein
effects are then averaged inside the sequence/domain union groups used to
construct the fixed split, and union groups receive equal weight. Confidence
intervals resample whole union groups. Two-sided sign-flip tests also operate
on union-group effects.

One primary Phase 2 family was declared:

- six model conditions;
- two contribution signs;
- RSA tested after matching on Q3 and Neq;
- torsional change, distance to the nearest Q3 boundary and MD strain tested
  after matching on Q3, Neq and RSA;
- 48 two-sided tests in total, with one Benjamini-Hochberg correction across
  all 48.

The remaining annotation tests are exploratory. Their upper, lower and
two-sided p-values are retained for diagnostics but are not additional primary
families.

Under the Q3 + Neq comparison, positive apices had RSA values 0.105–0.131
higher than matched controls. This was significant in all six models. Negative
apices had RSA values 0.019–0.120 lower than matched controls. The direction
was consistent across all six models, although the corrected test was
significant in four; the effect was weakest in the top-fine-tuned ESM3 models.

After additionally matching RSA, the union-group-weighted test-set effects
across the six models were:

| Sign | Torsional change | Distance to Q3 boundary | MD strain |
|:---|---:|---:|---:|
| Negative | -18.58 to -13.75 degrees | +1.91 to +2.85 residues | -0.0078 to -0.0042 |
| Positive | +15.77 to +28.31 degrees | +0.37 to +0.88 residues | +0.0042 to +0.0133 |

Torsional change remained significant in all six conditions for both signs.
Negative boundary distance was significant in all six models and negative
strain in five. Positive boundary distance was significant in four models and
positive strain in three. The position-matched results remain available as
a more restrictive sensitivity analysis but are not the primary estimates.

The raw circular-shift analysis in Phase 2B remains useful for showing that
apex locations are not arbitrary. Because the model predicts the Neq-derived
target accurately and \(s_j\) is closely related to its local decision
evidence, the broad raw enrichments for flexibility, exposure and Q3/Q8 state
are primarily model-localization diagnostics. The stricter matched torsion,
boundary and external-strain effects are the main biological results.

Current outputs:

```text
results/publication_comparable_v2/analysis_phase2_primary_no_position_test/
  enrichment_with_test_strain/
    within_q3_match_coverage_summary.csv
    within_q3_match_balance.csv
    within_q3_matched_cases.csv.gz
    within_q3_matched_controls.csv.gz
    within_q3_matched_effects_by_protein.csv.gz
    within_q3_matched_enrichment_summary.csv
    headline_union_group_inference.csv

  pipeline_audit_with_test_strain.json
```

The audit passed all 49 checks. The expensive Phase 2B circular-shift
null was reused without recomputation because the correction affected only
control matching and downstream inference, not apex locations or annotations.

### 2E. Test-set strain extension

Strain source:

```text
/home/zahralab/MDStrainMapper/results/atlas_grouped_v1_test
```

Expected layout:

<strain_root>/<protein>/strain_summary.csv

Required columns:

- residue
- ensemble_mean
- ensemble_std

Scope:

- strain is available only for the 208 test proteins
- it is never imputed into train or validation
- all 208 test files passed schema, length and exact residue-index checks
- the first and last 10 residues are excluded from strain analyses
- “high strain” is the top 10% of valid strain values within each protein

Derived strain metrics:

- strain_ensemble_mean
- strain_ensemble_std
- absolute strain change from the preceding residue
- top-strain-decile indicator
- distance to the nearest top-strain-decile residue

Final strain-aware output directory:

```text
results/publication_comparable_v2/
  analysis_phase2_primary_no_position_test/
    enrichment_with_test_strain/
```

Raw circular-shift results across the six model conditions:

| Property at apex | Positive observed | Positive null | Negative observed | Negative null |
|:---|---:|---:|---:|---:|
| Mean strain | 0.1044–0.1141 | 0.0772–0.0805 | 0.0578–0.0612 | 0.0748–0.0767 |
| Absolute strain gradient | 0.0279–0.0309 | 0.0194–0.0200 | 0.0143–0.0158 | 0.0189–0.0193 |
| Top-decile strain frequency | 25.56–30.51% | 10.22–10.30% | 1.79–3.07% | 10.21–10.34% |
| Distance to top-decile strain (residues) | 8.11–9.96 | 11.03–11.37 | 14.12–14.56 | 11.12–11.50 |

For every property in this table, observed values differed significantly from
the sign-matched circular-shift control in all six models. The two-sided
BH-adjusted p-values were at most 0.00636.

Same-protein Q3-only strain results:

Positive Q3-C apices versus Q3-C controls:

- mean strain difference: +0.0194 to +0.0275
- strain-gradient difference: +0.00680 to +0.00964
- top-decile strain enrichment: +11.68 to +17.18 percentage points
- all three effects were significant in all six conditions

Negative Q3-H apices versus Q3-H controls:

- mean strain difference: -0.0158 to -0.0113
- strain-gradient difference: -0.00417 to -0.00230
- top-decile strain difference: -6.36 to -4.90 percentage points
- mean strain and top-decile frequency were significant in all six conditions;
  the gradient difference was significant in four of six

Negative Q3-E apices versus Q3-E controls:

- mean strain difference: -0.0181 to -0.00294
- strain-gradient difference: -0.0117 to -0.00154
- top-decile strain difference: -4.00 to -1.43 percentage points
- the directions were consistent, but mean strain and top-decile frequency
  were significant in four of six conditions and the gradient in three of six

The negative Q3-E comparisons contain only 3–34 apices from 3–25 proteins per
model. They are therefore descriptive and do not establish a robust
cross-model strain effect for strand apices.

After Q3-only control, positive Q3-C apices were significantly closer to a
high-strain residue in four of six models, while negative Q3-H apices were
farther away in all six. The distance result was not significant for negative
Q3-E apices.

High-strain identifier results for positive bands:

| Measurement | Result across models |
|:---|---:|
| Positive apices in the high-strain decile | 25.56–30.51% |
| High-strain residues detected by an exact positive apex | 4.51–6.86% |
| Residues in positive-band intervals that are high-strain | 25.56–31.07% |
| High-strain residues covered by positive-band intervals | 19.86–30.49% |

Conclusion:

Positive apices are enriched for locally high and rapidly changing strain,
even relative to same-protein Q3-C controls. Negative structured apices show
the opposite pattern most consistently for Q3 helices. The negative Q3-E
sample is too small for an equally strong conclusion about strands.

However, most high-strain residues are not exact positive apices. Positive
bands are enriched markers of high-strain environments, not general
high-strain-residue detectors.

Because strain is currently test-only, these strain associations have not yet
been replicated on independent train/validation strain datasets.

### 2F. Outputs, visualization and audit

The authoritative current strain-aware outputs are:

```text
results/publication_comparable_v2/
  analysis_phase2_primary_no_position_test/
    enrichment_with_test_strain/
      apex_metrics_by_protein.csv.gz
      apex_circular_shift_null.csv.gz
      apex_circular_shift_enrichment_summary.csv
      paired_flex_vs_rigid_summary.csv
      annotation_band_coverage_by_protein.csv.gz
      annotation_band_coverage_identifier_summary.csv
      within_q3_matched_cases.csv.gz
      within_q3_matched_controls.csv.gz
      within_q3_match_coverage_summary.csv
      within_q3_match_balance.csv
      within_q3_matched_effects_by_protein.csv.gz
      within_q3_matched_enrichment_summary.csv
      headline_union_group_inference.csv
      strain_input_audit.csv
      biophysical_enrichment_parameters.json
    pipeline_audit_with_test_strain.json
```

Visualization script:

```text
signed_band_analysis/plot_signed_band_phase2_results.py
```

The existing figures predate the matching correction and are not authoritative
for corrected p-values or matched-effect labels:

```text
results/publication_comparable_v2/
  analysis_phase2_interval_iou05_test/phase2_figures/
```

These include:

- Q3 and Q8 composition plots
- continuous Neq, RSA and torsion compositions
- Q3-only matched effect plots
- inverse selection/identifier plots
- matching-control availability diagnostics
- stricter matching diagnostics

Publication figures should be regenerated from the corrected directory.

Pipeline scripts:

```text
signed_band_analysis/run_signed_band_biophysical_pipeline.sh
```

```text
signed_band_analysis/audit_signed_band_biophysical_pipeline.py
```

Final corrected audit:

```text
results/publication_comparable_v2/
  analysis_phase2_primary_no_position_test/
    pipeline_audit_with_test_strain.json
```

- passed
- 49 checks
- 0 failures
- all 208 test strain files valid
- controls confirmed outside all band intervals
- Q3 matches and matching calipers independently verified
- summary statistics independently recomputed
- matching covariates absent from inferential outcome rows
- one 48-test primary p-value family confirmed

## PHASE 3A: WHY SOME Q8 SEGMENTS ARE SELECTED (only exploratory, not to be used as a main result)

Script:

```text
signed_band_analysis/analyze_signed_band_object_selection.py
```

Phase 2 describes the residue found at a band apex. Phase 3A asks a different
question: Do Q8 segments containing contribution-band apices differ from same-protein,
same-Q8 segments that contain no band?

Phase 3A compares Q8 segments that contain a seed-stable band apex with Q8
segments that do not. It then tests whether their ATLAS, NetSurfP and sequence
features differ.


### 3A.1 Cases and controls

The analysis divides each protein into complete contiguous Q8 segments.

A positive case:

- contains at least one positive apex;
- contains no negative apex.

A negative case:

- contains at least one negative apex;
- contains no positive apex.

A clean control:

- is in the same protein and has the same Q8 label as a case;
- contains no apex;
- does not overlap any positive or negative band interval.

Segments containing both signs are excluded. A segment that overlaps a band
but does not contain its apex is also excluded rather than treated as a clean
control. The statistical comparisons retain only protein/Q8 groups containing
both a case and a clean control.

Neq, RSA, segment length, position and the other explanatory features are not
used to choose controls. This allows the analysis to test whether they differ
between selected and unselected segments.

### 3A.2 Inputs and feature sources

The current run uses the upgraded Phase 1 seed-stable catalog for all three
data splits and the residue annotation table prepared in Phase 2.

The feature sources are:

| Feature group | Measurements | Source |
|:---|:---|:---|
| Q8 identity | Q8 segment label | NetSurfP prediction |
| Neq | Mean Neq, maximum Neq, Neq-peak fraction and Neq-peak excess | ATLAS molecular dynamics |
| Exposure, length and position | Mean and maximum RSA, log segment length and normalized midpoint | NetSurfP RSA and sequence coordinates |
| Geometry and boundaries | Mean and maximum torsional change, Q3-boundary fraction, distance to a Q3 boundary and structured-linker fraction | Derived from NetSurfP phi/psi and Q3/Q8 predictions |
| Disorder | Mean and maximum disorder score | NetSurfP prediction |
| Sequence composition | Glycine, proline, hydrophobic, charged and aromatic fractions, plus sequence entropy | Protein sequence |

The geometry in Phase 3A is therefore predicted geometry obtained from NetSurfP. Experimental PDB
coordinates are not used until Phase 3B.

### 3A.3 Current object counts

The latest candidate table contains 481,842 rows across six model conditions
and the train, validation and test splits. These are analysis rows, not unique
biological segments: the Q8 segmentation for a protein is repeated for each
model condition.

The mutually exclusive categories are:

- 44,100 positive-only selected segments;
- 27,095 negative-only selected segments;
- 262 segments containing both signs;
- 355,689 clean controls;
- 54,696 segments that overlap a band but contain no apex.

The test subset contains 73,044 rows: 6,703 positive-only cases, 4,128
negative-only cases, 44 dual-sign segments, 53,842 clean controls and 8,327
overlap-without-apex segments.

Every stable Phase 1 apex is assigned to a Q8 segment. The much smaller
overlap-without-apex category compared with the old analysis reflects the new,
narrower, nonoverlapping Phase 1 bands.

### 3A.4 Sequential selection models

For each ESM condition and sign, the script fits a separate logistic-regression
analysis model.

The feature stages are cumulative:

1. Q8 only.
2. Add Neq.
3. Add RSA, segment length and position.
4. Add predicted torsional geometry and Q3 boundaries.
5. Add predicted disorder.
6. Add coarse amino-acid composition.

The logistic-regression models are fitted only on train proteins. The fitted
models are then applied without refitting to validation and test proteins.
Cases and controls receive equal total weight within each protein/Q8 group.

Mean weighted AUROC across the six ESM conditions is:

| Feature stage | Positive validation | Positive test | Negative validation | Negative test |
|:---|---:|---:|---:|---:|
| Q8 only | 0.500 | 0.500 | 0.500 | 0.500 |
| Add Neq | 0.758 | 0.751 | 0.733 | 0.757 |
| Add RSA, length and position | 0.814 | 0.809 | 0.840 | 0.848 |
| Add predicted geometry and boundaries | 0.836 | 0.834 | 0.851 | 0.857 |
| Add predicted disorder | 0.841 | 0.837 | 0.853 | 0.858 |
| Add sequence composition | 0.844 | 0.838 | 0.857 | 0.863 |

The Q8-only AUROC is 0.5 by design because cases and controls are compared
within the same Q8 type and are balanced within those groups. It should not be
interpreted as evidence that secondary structure is generally unrelated to
band location.

The similar validation and test results show that the segment-selection
patterns learned from train proteins transfer to both held-out splits.

### 3A.5 What the results show

Neq provides the first large improvement, but Neq alone does not explain which
same-Q8 segment is selected. RSA, length and position provide another large
improvement. Predicted geometry and boundary features add more information,
particularly for positive segments. Disorder and coarse sequence composition
provide only small additional improvements after the earlier features.

The direct within-protein, same-Q8 comparisons show that:

- positive cases occur mainly in Q8 C, T and S segments;
- selected positive C, T and S segments generally have higher Neq, greater
  exposure, greater length, higher disorder and fewer internal Q3 boundaries
  than their same-Q8 controls;
- torsion depends on the Q8 subtype: selected T segments have greater mean
  torsional change, selected C segments have lower mean torsional change, and
  the S-segment effect is weak;
- negative cases occur mainly in Q8 H segments;
- selected negative H segments are longer and have lower Neq, lower disorder,
  lower mean torsional change and fewer internal Q3 boundaries than unselected
  H segments in the same protein;
- negative E segments are uncommon in the new catalog and should remain
  descriptive rather than support a general strand conclusion.

The C-segment torsion result is not the same comparison as the Phase 2 apex
result. Phase 2 measures torsion at one selected residue after Q3 matching;
Phase 3A averages torsion across an entire Q8 C segment and compares it with
other complete Q8 C segments.

These results identify reproducible associations with model selection. They do
not show that the features cause selection, and the NetSurfP variables are
predictions rather than experimental structural measurements.

Feature-effect rows supported by fewer than 10 proteins remain descriptive and
receive no confidence interval, p-value or adjusted p-value.

### 3A.6 Outputs

```text
results/publication_comparable_v2/
  analysis_phase3a_interval_iou05_stable/
    q8_segment_candidates.csv.gz
    matched_feature_effects_by_protein.csv.gz
    matched_feature_effect_summary.csv
    sequential_model_performance.csv
    sequential_model_coefficients.csv
    phase3a_parameters.json
```

## PHASE 3B: COMPARISON WITH EXPERIMENTAL STRUCTURE

Phase 3B asks whether contribution-band locations have distinctive geometry or
contact environments in experimental PDB structures. It now contains three
analyses:

1. A direct comparison at the band apex. This is the primary analysis.
2. A smaller sensitivity analysis using the complete detected band intervals.
3. A Q8-segment analysis inherited from Phase 3A. This is a secondary analysis
   because the model does not select complete Q8 segments.

### 3B.1 Structure mapping

Experimental structures and C-alpha contact maps are prepared with:

```text
Attention/build_contact_maps_from_pdb.py
```

PDB residues are aligned explicitly to the model sequence. PDB residue numbers
are never assumed to equal model residue indices, and unresolved residues are
recorded rather than silently filled in.

Of the 1,383 proteins, 1,379 passed the mapping requirements. Every validation
and test protein passed. Three train proteins had less than 80% sequence
coverage, and one train structure could not be downloaded.

The experimental features include:

- C-alpha curvature and virtual torsion;
- contact degree and local packing;
- mean spatial distance between contacts;
- betweenness and closeness centrality;
- contact-community participation and boundary status;
- ECOD-domain boundary and cross-domain-contact measurements.

### 3B.2 Direct apex-to-structure analysis

Script:

```text
signed_band_analysis/analyze_signed_band_apex_structure.py
```

This analysis uses each seed-stable test-set band apex directly. Q8 is not used
to define the analysis object.

Each structurally resolved apex is compared with residues from the same protein
that lie outside every positive and negative band. Matching now uses the same
four control sets as Phase 2:

1. `q3_only`: same Q3 label.
2. `q3_neq`: same Q3 label and Neq within 0.25.
3. `q3_neq_rsa`: same Q3 label, Neq within 0.25 and RSA within 0.15.
4. `q3_neq_rsa_position`: the preceding requirements plus normalized sequence
   position within 0.25.

The Q3-only comparison uses all eligible controls. The other comparisons use
the five nearest eligible controls. Position is not part of the initial
matching. It is included only in the final sensitivity analysis to check
whether terminal location explains an effect. Exact distance ties use the same
reproducible randomized tie-break as Phase 2 rather than residue order.

The statistics also follow Phase 2. Apex-minus-control effects are averaged
within protein and then within the fixed-split union groups. Union groups
receive equal weight, are resampled for confidence intervals and are the units
of the two-sided sign-flip tests. The primary structural family contains 60
Q3 + Neq-matched tests: six conditions, two signs and five predeclared contact
features. Benjamini-Hochberg correction is applied once across those 60 tests.
Q3 + Neq + RSA asks whether an effect remains among equally exposed residues,
and the position-matched analysis is retained as a final sensitivity check.
The larger protein-level feature table remains exploratory.

The audit passed. The six model catalogs contain 11,039 test-set apices, of
which 10,804 have mapped PDB coordinates. Control coverage was:

| Matching | Negative apices | Positive apices |
|:---|---:|---:|
| Q3 only | 99.86–100% | 95.34–97.56% |
| Q3 + Neq | 99.86–100% | 88.46–90.38% |
| Q3 + Neq + RSA | 99.85–100% | 79.10–80.93% |
| Q3 + Neq + RSA + position | 99.44–100% | 64.48–67.84% |

The positive-apex pattern was strong. After Q3 + Neq matching, positive apices
had 1.03–1.37 fewer contacts and a 0.17–0.23 lower packing index. All five
contact-network features were significant in all six models. After also
matching RSA, positive apices still had 0.49–0.83 fewer contacts and a
0.09–0.14 lower packing index, significant in all six models. Their weakly
connected structural environment is therefore not explained only by greater
solvent exposure.

For negative apices, the Q3 + Neq results were model-dependent. Contact degree
and packing were significant in three of six models, betweenness and closeness
in four, and participation coefficient in all six. After RSA matching,
contact degree was slightly lower in all six models rather than higher, while
packing and centrality were mostly weak or inconsistent. Negative apices are
buried, but the evidence does not support an additional universal hub property
beyond what is expected from burial.

The same script tests whether stronger bands, measured by
`log2(R_p / 2)`, have stronger structural signatures. After protein adjustment
and multiple-testing correction, these strength associations were weak and
not consistent across all six models. The main result is therefore the
location of positive and negative apices, not a universal relationship between
band rank and structural-effect size.

Primary outputs:

```text
results/publication_comparable_v2/
  analysis_apex_structure_primary_no_position_test/
    stable_apex_structure_features.csv.gz
    matched_nonband_controls.csv.gz
    matched_apex_control_effects_by_protein.csv.gz
    matched_apex_control_effect_summary.csv
    headline_union_group_network_inference.csv
    importance_R_structure_X_associations.csv
    match_coverage_balance.csv
    structural_feature_coverage.csv
    run_audit.json
```

The corrected structural run passed its internal audit.

#### Full-band interval sensitivity

The apex is only one residue within a detected band. A smaller sensitivity
analysis therefore repeats the comparison using each actual detected band
interval:

```text
signed_band_analysis/analyze_signed_band_interval_structure.py
```

Each control is a same-protein, non-band interval with the same width and the
same apex-to-left-boundary and apex-to-right-boundary distances as the detected
band. Matching uses the same four schemes. Inference uses protein-level sign
flips, protein bootstrap confidence intervals and Benjamini-Hochberg
correction. The main cohort requires every residue in both the band and control
interval to have a mapped C-alpha coordinate. An 80%-resolved cohort is
retained as a comparison.

Under the fully resolved, strictest matching, the positive-band intervals had
0.82–1.25 fewer contacts, lower packing and centrality, and 0.08–0.17 Å greater
mean contact distance than controls. These effects were significant in all six
models. Negative-band intervals had 0.39–0.66 more contacts, greater packing
and centrality, 0.11–0.17 Å shorter mean contact distance, greater curvature
and lower absolute virtual torsion. Nearly all of these effects were significant
in all six models.

This resolves an important difference between analysis units: the exact
negative apex is not consistently a high-degree hub after strict residue-level
matching, but the larger negative-band interval lies in a more connected and
more tightly packed structural environment. The positive low-connectivity
result is consistent at both the apex and interval levels.

Outputs:

```text
results/publication_comparable_v2/
  analysis_band_interval_structure_phase2_consistent_test/
    band_interval_structure_features.csv.gz
    matched_nonband_intervals.csv.gz
    band_interval_match_coverage.csv
    matched_band_interval_effects_by_protein.csv.gz
    matched_band_interval_effect_summary.csv
    structure_mapping_audit.csv
    run_audit.json
```

### 3B.3 Q8-segment analysis

Script:

```text
signed_band_analysis/analyze_signed_band_external_structure.py
```

This analysis attaches experimental structure to the Phase 3A objects. A case
is a complete Q8 segment containing a positive or negative apex. A control is a
same-protein, same-Q8 segment that does not overlap any band.

The unit is therefore a Q8 segment containing an apex, not a segment selected
by the model. Segment averages can dilute localized effects, segment length
affects the chance of containing an apex, and the Q8 boundaries come from
NetSurfP. For these reasons, this analysis is best treated as a secondary or
supplementary Q8-stratified analysis.

Small logistic-regression models were trained on train proteins and applied
without refitting to validation and test proteins. Mean test AUROC across the
six ESM conditions was:

| Feature stage | Positive | Negative |
|:---|---:|---:|
| Q8 only | 0.500 | 0.500 |
| Add Neq, RSA, length and position | 0.807 | 0.846 |
| Add NetSurfP torsion and boundaries | 0.830 | 0.856 |
| Add experimental PDB geometry | 0.837 | 0.862 |
| Add contact-network features | 0.846 | 0.865 |
| Add ECOD-domain features | 0.852 | 0.866 |

Validation performance was similar: final mean AUROC was 0.852 for positive
segments and 0.856 for negative segments.

The matched segment comparisons found that positive C, T and S segments
containing apices generally had higher test-only MD strain and lower contact
connectivity than same-Q8 controls. Negative H segments containing apices had
lower strain and modestly greater connectivity. Negative E samples were too
small for a general strand conclusion. These results are associations with
apex-containing structural context, not evidence that the model selected a
whole Q8 segment.

Outputs:

```text
results/publication_comparable_v2/
  analysis_phase3b_interval_iou05_stable/
    external_features_by_q8_segment.csv.gz
    matched_external_effects_by_protein.csv.gz
    matched_external_effect_summary.csv
    sequential_external_model_performance.csv
    mechanism_class_external_associations.csv
    mechanism_band_external_features.csv.gz
    structure_mapping_audit.csv
    parameters.json
```

Strain is available only for test proteins. It was used for matched descriptive
comparisons but was not included in models trained on train proteins. The joins
with Phase 3C mechanism classes are exploratory and inherit the Q8-segment
limitations.

### 3B.4 Structural safeguards

Script:

```text
signed_band_analysis/analyze_phase3b_structural_safeguards.py
```

Two safeguards were added to the Q8-segment analysis.

First, the main comparisons were repeated in stricter structure cohorts:

- the reference cohort requires at least 80% protein mapping and 80% of the
  segment to be resolved;
- the fully resolved local cohort requires every segment residue to have a
  mapped C-alpha coordinate;
- the fully resolved geometry cohort requires every coordinate needed for the
  PDB geometry measurements;
- the strict graph cohort additionally requires complete contact-graph nodes
  and at least 95% protein mapping.

Cases and controls were filtered symmetrically, and protein/Q8 groups lacking a
valid case or control after filtering were removed. For the dominant Q8 types,
the fully resolved local and strict graph analyses preserved the direction of
every central geometry and contact effect. The fully resolved geometry analysis
preserved 161 of 168 model/Q8/feature directions. Missing PDB coordinates
therefore do not explain the main segment-level directions.

Second, uncertainty in each AUROC increase was measured by resampling whole
test proteins 2,000 times. Every selected segment and control from a sampled
protein was kept together, and the smaller and larger models were evaluated on
the same bootstrap sample.

For the reference cohort, the mean AUROC increases across six ESM conditions
were:

| Added information | Positive change (95% interval) | Negative change (95% interval) |
|:---|---:|---:|
| Experimental PDB geometry | +0.0067 (0.0030 to 0.0107) | +0.0067 (0.0030 to 0.0103) |
| Contact network | +0.0093 (0.0064 to 0.0122) | +0.0030 (0.0001 to 0.0059) |
| ECOD/domain | +0.0054 (0.0039 to 0.0070) | +0.0009 (-0.0007 to 0.0024) |
| NetSurfP geometry plus all external structure | +0.0443 (0.0370 to 0.0519) | +0.0203 (0.0139 to 0.0268) |

The total increase from all stages after base biophysics remained positive in
the fully resolved geometry cohort: +0.0416 for positive segments and +0.0237
for negative segments, with both 95% intervals excluding zero. However, the
isolated positive PDB-geometry increase became uncertain in that strict cohort
(+0.0019; 95% interval -0.0036 to 0.0073). Contact and domain features carried
most of the remaining positive increment. The negative PDB-geometry increment
remained positive, while its contact and ECOD increments individually included
zero.

All safeguard audits passed, including candidate provenance, coordinate
completeness, saved-prediction reconstruction and synchronized protein
bootstrapping.

Safeguard outputs:

```text
results/publication_comparable_v2/
  analysis_phase3b_structural_safeguards_interval_iou05_stable/
    resolution_sensitivity/
    bootstrap/
    audits/
    README.md
    parameters.json
```

### 3B.5 Interpretation and limitations

The direct apex analysis is the clearest Phase 3B result because it studies the
actual model-derived location and does not require a Q8 segment to be the
biological object. It supports a reproducible association between positive
apices and weakly connected, loosely packed experimental structure.

The negative-apex result is more complicated and should not be summarized as a
universal packed-core or hub signal after Neq/RSA matching. The Q8-segment
analysis provides useful multivariable and sensitivity checks, but its object
definition is imposed after model inference and should remain secondary.

All results remain associative. PDB structures are incomplete for some flexible
regions, ECOD boundaries are annotations rather than measured hinge axes, and
contact-network results depend on the contact definition. Neither analysis
establishes a causal allosteric or communication pathway.

## PHASE 3C: INTERNAL EVIDENCE/CONSULTATION MECHANISM

Script:

```text
signed_band_analysis/analyze_signed_band_model_mechanism.py
```

Scientific questions:

1. Does a band apex contain unusually strong signed evidence, `s_j`?
2. Is the apex consulted unusually strongly by the rest of the protein, through
   its mean incoming attention, `B_j`?
3. Are both components elevated at the same apex?
4. Is their relative importance different for positive and negative bands?

### 3C.1 Matched control design

For each band apex, controls are:

- in the same protein
- in the eligible protein interval
- exact same Q8 subtype as the apex
- outside every positive and negative band interval

The control value is the mean over all eligible same-protein, same-Q8 control
residues.

This is a residue-level comparison. Q8 is used only to choose comparable
control residues; the analysis object remains the model-derived band apex. It
is therefore different from Phase 3A, where the whole Q8 segment is the object.

### 3C.2 Seed-aware exact decomposition

Bands were detected from the three-seed mean influence profile, but the
identity

    I_j = s_j * B_j

holds exactly within each individual seed.

Therefore, for each seed separately:

    delta log|I|
      = delta log|s|
      + delta log(B)

For each seed, the script:

1. reads `s_j`, `B_j` and `I_j` at the band apex;
2. calculates the corresponding mean over the matched control residues;
3. calculates the apex-minus-control difference for each component;
4. verifies the exact log decomposition;
5. averages the three seed-specific effects only after these calculations.

This avoids incorrectly multiplying seed-averaged s_j and seed-averaged B_j.

For the matched summaries, band effects are first averaged within each protein,
so a protein with many bands does not receive more weight. The current script
reports a 95% t-based confidence interval across proteins, tests the
protein-level effects with a two-sided Wilcoxon signed-rank test, and applies
Benjamini-Hochberg correction across the reported mechanism tests.

### 3C.3 Mechanism classes

The classes describe which component produces the increase in `|I_j|` relative
to matched controls. For an apex with positive total log-magnitude enrichment:

- evidence-dominated:
  evidence supplies at least two-thirds of the increase, or evidence increases
  while consultation does not;

- consultation-dominated:
  consultation supplies at least two-thirds, or consultation increases while
  evidence does not;

- combined:
  both increase and neither supplies more than two-thirds.

Additional classes:

- not_magnitude_enriched:
  the apex does not have greater `|I_j|` than its matched controls;

- unclassified:
  there is no eligible exact-Q8 control, or the decomposition does not support
  one of the preceding classes.

### 3C.4 Coverage and seed stability

The current run covers train, validation and test. The numerical results below
use the held-out test subset of the final seed-stable Phase 1 catalog. Each
model contributes its own band catalog:

| Model | Negative | Positive | Total |
|:---|---:|---:|---:|
| ESM2 frozen | 653 | 1,059 | 1,712 |
| ESM2 top 4 | 713 | 1,227 | 1,940 |
| ESM2 top 28 | 665 | 1,194 | 1,859 |
| ESM3 frozen | 752 | 1,115 | 1,867 |
| ESM3 top 4 | 707 | 1,086 | 1,793 |
| ESM3 top 28 | 750 | 1,118 | 1,868 |

These are not 11,039 unique biological regions. They are 11,039 model-specific
band records across six separate catalogs.

Exact-Q8 controls were available for 10,958 of the 11,039 records. The 81
without a control remain in the per-band table as unclassified but do not enter
the matched effect summaries. Of these, 77 were positive and four were
negative.

Seed-direction agreement was nearly complete:

- 11,038 bands had the same influence direction in all three seeds;
- one band agreed in two of three seeds;
- no band agreed in only one seed.

### 3C.5 Positive-band mechanism results

Median apex-versus-control effects across the six models were:

- `delta log|s| = 1.20`, corresponding to 3.31-fold stronger intrinsic
  evidence;

- `delta log(B) = 1.06`, corresponding to 2.89-fold greater attention breadth;

- `delta log|I| = 2.27`, corresponding to 9.66-fold greater total influence.

All three effects were positive and significant after Benjamini-Hochberg
correction in all six models.

The mean band fractions across the six model catalogs were:

- combined: 90.23%;
- evidence-dominated: 7.91%;
- consultation-dominated: 0.71%;
- not magnitude-enriched: 0.02%;
- unclassified: 1.13%.

Interpretation:

Positive band apices usually combine:

- strong intrinsic flexibility-supporting evidence;
- increased consultation by other residues.

Neither component alone explains most positive bands in the updated catalog.

### 3C.6 Negative-band mechanism results

Median apex-versus-control effects across the six models were:

- `delta log|s| = 1.21`, corresponding to 3.35-fold stronger intrinsic
  evidence;
- `delta log(B) = 0.75`, corresponding to 2.12-fold greater attention breadth;
- `delta log|I| = 1.95`, corresponding to 7.04-fold greater total influence.

All three effects were positive and significant after Benjamini-Hochberg
correction in all six models.

The mean band fractions across the six model catalogs were:

- combined: 71.61%;
- evidence-dominated: 28.27%;
- consultation-dominated: 0.02%;
- not magnitude-enriched: 0%;
- unclassified: 0.10%.

Interpretation:

Most negative bands are also combined evidence-plus-consultation objects in the
updated catalog. However, negative bands are more evidence-heavy than positive
bands: 28.27% were evidence-dominated, compared with 7.91% of positive bands,
and their median attention-breadth enrichment was smaller. ESM2 frozen was the
only model in which evidence-dominated negative bands slightly outnumbered
combined negative bands.

Purely consultation-dominated negative bands were exceptionally rare.

### 3C.7 Audited numerical identities

- the 11,039 band identifiers exactly match the final Phase 1 stable test
  catalog;

- the mean of the three seed `I_j` values reproduces the stored averaged apex
  `I_j` with maximum error `1.11e-16`;

- the maximum directly checked `I_j - s_j B_j` error was `3.29e-8`;

- the maximum within-seed log-decomposition error was `3.91e-7`, and the
  maximum error after seed averaging was `2.03e-7`;

- all 18 expected test cache files were present:

      6 models x 3 seeds

- all 33,117 expected band/seed rows were present:

      11,039 model-specific bands x 3 seeds

Conclusion:

Both positive and negative bands usually combine intrinsic signed evidence with
learned attention routing. The difference is quantitative rather than absolute:

- positive bands show the stronger attention-breadth enrichment and are almost
  always classified as combined;
- negative bands are still usually combined, but a substantially larger
  fraction are evidence-dominated.

This analysis explains how the observed influence at detected apices is divided
between `s_j` and `B_j`. It does not ask whether the same regions would have
been detected if attention were uniform; that separate counterfactual question
is addressed by the Phase 1 uniform-attention control.

### 3C.8 Outputs

```text
results/publication_comparable_v2/
  analysis_phase3c_interval_iou05_stable_all_splits/
    mechanism_by_band_and_seed.csv.gz
    mechanism_by_band_seed_averaged.csv.gz
    mechanism_effects_by_protein.csv.gz
    mechanism_effect_summary.csv
    mechanism_class_summary.csv
    per_seed_split_cache/
    phase3c_parameters.json
```


## PHASE 4: QUERY RECEIVERS OF SIGNED BAND CONTRIBUTIONS

Phase 4 asks why, for one fixed source band, some query residues receive a
large sign-aligned contribution while other queries in the same protein
receive little.

For query residue i and source band b:

    R_i,b = sign_b * sum_(j in band b) C_ij

where larger R_i,b means that query i receives a stronger contribution in the
direction represented by the source band.

For one fixed key:

    C_ij = s_j * A_ij

Because s_j is constant across queries, query-to-query variation for that key
comes from attention A_ij. For a multi-residue band, the receiver profile sums
the contributions from all keys in the band.

### 4.1 Scripts and current inputs

Script:

```text
signed_band_analysis/analyze_signed_band_query_receivers.py
signed_band_analysis/build_band_query_structural_water_features.py
```

The upgraded run uses the all-split contribution manifest, the current
seed-stable Phase 1 catalog, Phase 2 residue annotations, the corrected
all-split Phase 3C mechanism table, alignment-audited experimental structures
and ECOD annotations.

```text
results/publication_comparable_v2/
  all_split_signed_contributions_manifest.tsv
  analysis_phase1_upgraded_raw_mad2/
    reproducibility_interval_iou05/stable_signed_bands.csv
    mean/signed_band_protein_summary.csv
  analysis_phase2_upgraded_raw_mad2/annotations/
    residue_biophysical_annotations.csv.gz
  analysis_phase3c_interval_iou05_stable_all_splits/
    mechanism_by_band_seed_averaged.csv.gz
```

The Phase 1 catalog path recorded by this run is byte-identical to the current
canonical catalog under `reproducibility_interval_iou05_null_fixed/`.

Query features:

- Q8 subtype
- Neq
- RSA
- disorder
- torsional change
- coarse amino-acid class
- normalized sequence position
- whether the query lies inside any signed band

Experimental pair features include 3D geometry, backbone orientation, direct
C-alpha contacts, ordinary and nonlocal contact-network paths, contact
communities, ECOD relationships, putative polar contacts and
crystallographic-water paths.

A putative polar contact is an N/O/S heavy-atom separation of at most 3.5 Å;
it is not labelled as a hydrogen bond. The primary water analysis uses
quality-filtered waters from X-ray structures at resolution at most 2.5 Å,
with occupancy at least 0.5, 3.4 Å protein-water contacts and 3.2 Å
water-water contacts. A water path contains no intervening protein nodes and
is a static geometric compatibility measure, not solution-state persistence.

### 4.2 Receiver profiles and matching

The script streams each LxL matrix and calculates, for every source-band/query
pair:

- raw signed contribution sum
- sign-aligned directional contribution R_i,b
- contribution per band residue
- fraction of the query’s absolute contribution supplied by the band
- apex-key contribution
- summed and mean attention to the band
- attention to the apex

Metrics are calculated within each seed and then averaged across the three
seeds.

Within each source band and protein:

- high receiver: R_i,b at or above the 90th percentile
- low receiver: R_i,b at or below the 50th percentile
- middle receivers are excluded from high-versus-low analysis

High and low receivers are matched on:

- same protein
- same source band
- exact query Q8 subtype
- same distance-to-band bin

Distance bins:

- inside the source band
- 1–5 residues
- 6–20 residues
- 21–50 residues
- more than 50 residues

Effects are first calculated within band/Q8/distance strata and then aggregated
with equal protein weighting.

### 4.3 Data coverage

All six conditions and all three fixed splits were analyzed separately. The
upgraded receiver tables contain 24,665,261 band-query pairs:

- ESM2 frozen: 3,880,584
- ESM2 top4: 4,263,287
- ESM2 top28: 4,005,525
- ESM3 frozen: 4,165,284
- ESM3 top4: 4,112,398
- ESM3 top28: 4,238,183

The structural feature store contains 24,642,897 eligible pair rows in 8,208
atomic protein/model/split partitions. Missing coordinates are stored as
missing values, never as zero distance or no contact.

For predictive modeling, high and low receivers are balanced within each
band/Q8/distance stratum, with at most 50 rows per receiver class and protein.

### 4.4 Held-out receiver models

Logistic models were trained only on train proteins and evaluated without
refitting on validation and test proteins.

Stages:

1. Distance/Q8 baseline:
   - query Q8
   - distance bin
   - exact sequence separation
   - N-terminal versus C-terminal direction
   - whether the query lies inside the source band

2. Add query biophysical features.

3. Add experimental 3D geometry.
4. Add contact-network features.
5. Add community and ECOD-domain features.
6. Add putative polar-contact geometry.
7. Add crystallographic-water features on a fixed water-eligible cohort.
8. Add source-mechanism interactions.

On the test set, the baseline AUROC was 0.496–0.517 for negative sources and
0.499–0.527 for positive sources. Adding query biophysics increased AUROC to
0.951–0.976 and 0.865–0.957, respectively.

The baseline is near chance because high and low examples were deliberately
balanced within Q8 and distance strata. The large improvement therefore comes
from query biophysics rather than simple sequence proximity or Q8 identity.

Source-mechanism interactions changed test AUROC by -0.0003 to +0.0053 for
negative sources and -0.0010 to +0.0007 for positive sources. They therefore
add little discrimination once query properties are known.

### 4.5 Feature importance

Test AUROC changes across the six conditions:

| Feature | Negative: single / unique | Positive: single / unique |
|---|---:|---:|
| Query lies in any band | +0.299 to +0.405 / +0.031 to +0.050 | +0.247 to +0.401 / +0.062 to +0.140 |
| Neq | +0.287 to +0.341 / +0.006 to +0.025 | +0.249 to +0.293 / +0.015 to +0.041 |
| Torsional change | +0.370 to +0.412 / +0.008 to +0.023 | +0.066 to +0.085 / +0.001 to +0.003 |
| RSA | +0.156 to +0.188 / approximately zero | +0.108 to +0.170 / +0.006 to +0.019 |
| Disorder | +0.195 to +0.253 / approximately zero | -0.012 to +0.016 / +0.004 to +0.016 |
| Amino-acid class | +0.081 to +0.123 / approximately zero | +0.062 to +0.097 / +0.0004 to +0.0018 |
| Normalized position | approximately zero | approximately zero |

“Single” is the gain from adding one feature to the baseline.

“Unique” is the loss from removing that feature from the full query model.

Query-in-band status is the largest unique predictor, followed by Neq and, for
negative receivers, torsional change. Query-in-band status comes from the same
model contribution catalog, so it describes internal organization rather than
independent biology. Amino-acid class and normalized position provide almost
no unique information.

### 4.6 Matched high-versus-low effects

After exact query-Q8 and distance-bin matching:

Positive-band high receivers versus low receivers:

- Neq: +0.633 to +0.797
- RSA: +0.096 to +0.129
- torsional change: +15.7 to +23.1 degrees
- probability of lying in any band: +39.6 to +72.7 percentage points
- normalized position: small and not consistently significant

Negative-band high receivers versus low receivers:

- Neq: -0.402 to -0.363
- RSA: -0.156 to -0.116
- torsional change: -46.6 to -41.3 degrees
- probability of lying in any band: +49.4 to +68.1 percentage points
- normalized position: small and not consistently significant

Conclusion:

- Positive bands preferentially influence flexible, exposed and torsionally
  changing queries.
- Negative bands preferentially influence rigid, buried and torsionally stable
  queries.
- High receivers are more likely to lie in another band interval, but shared
  flexibility class and model-derived band overlap can explain much of this
  association. It is not evidence of a discrete signaling network.

### 4.7 Experimental structure and water

After matching query Q8 and sequence-distance bin, high and low receivers show
structural associations:

- high negative-band receivers were 2.22–2.77 Å closer to the source band and
  5.4–13.0 percentage points more likely to make a direct C-alpha contact;
- high positive-band receivers were 0.30–1.12 Å farther away on average, but
  were still 1.6–9.1 percentage points more likely to make a direct C-alpha
  contact;
- direct putative polar contacts were 1.6–6.0 percentage points more frequent
  for negative receivers and 2.0–8.1 points more frequent for positive
  receivers.

Direct-contact and polar-contact directions were consistent across models.
Community and ECOD co-membership were weak or inconsistent. One-water bridges
and longer crystallographic-water paths were not consistent across models.

These associations add little held-out discrimination beyond query
biophysics. On the fixed all-query water-eligible cohort, ordinary structure
added 0.0012–0.0028 AUROC for negative sources and 0.0012–0.0072 for positive
sources. Adding water changed AUROC by -0.0006 to +0.0001 and -0.0008 to
+0.0011, respectively. Structural and water increments were also small and
inconsistent for queries at least 21 residues away, including after direct
C-alpha contacts were removed.

Phase 4 therefore shows that strong receivers can have spatial and chemical
associations with their source bands. It does not establish a general
contact-network or water-mediated pathway, and query biophysics explains
nearly all held-out discrimination.

### 4.8 Long-range receivers

Long range is defined as at least 21 residues from the source-band interval.

Fraction of test high receivers at long range:

| Source mechanism | Negative source | Positive source |
|---|---:|---:|
| Combined | 57.5–62.7% | 60.9–66.7% |

Mean fraction of directional contribution mass at long range:

- negative combined: 65.9–70.8%
- positive combined: 72.4–75.0%

A majority of high receivers and directional contribution mass can therefore
be distant in sequence.

These are absolute fractions, not enrichment relative to the number of
available long-range query residues. Long sequence distance is not equivalent
to long spatial distance.

### 4.9 Outputs and audit

Output root:

```text
results/publication_comparable_v2/
  analysis_phase4_interval_iou05_stable_query_receivers_upgraded/
    feature_store_primary/
    primary/<condition>/
    audit_primary_features.json
    audit_primary_complete.json
```

Important outputs:

- band_query_pair_manifest.csv
- receiver_feature_effects_by_protein.csv.gz
- receiver_feature_summary.csv
- receiver_model_performance.csv
- receiver_feature_ablation_performance.csv
- receiver_structural_water_ablation_performance.csv
- long_range_receiver_summary.csv
- receiver_complete.json
- parameters.json

The complete audit passed 998,281 checks across 8,208 feature partitions and
all six receiver output directories, with no stored failures.

### 4.10 Conclusion and limitations

The model does not distribute a band’s contribution uniformly across queries.
After controlling for source band, query Q8 and sequence distance, query
biophysics strongly identifies high receivers.

Query-in-band status is the largest unique predictor, followed by Neq and
sign-specific local biophysics. Positive bands preferentially influence
flexible and exposed queries, whereas negative bands preferentially influence
rigid and buried queries. Much of the influence is long range in sequence.

Limitations:

- Query-in-band status and receiver strength both come from the model's
  contribution decomposition; their association is not independent biology.
- Experimental geometry, contacts and putative polar contacts are independent
  annotations, but their incremental predictive value is small.
- Crystallographic waters describe one static structure and do not measure
  solution-state occupancy or transmission.
- Long sequence distance does not imply long spatial distance, although the
  upgraded analysis separately examines direct 3D contact.
- The six model conditions use the same proteins and are not independent
  biological datasets.
- The results are predictive associations and do not establish causal
  biochemical transmission.


## PHASE 5: LEGACY SIGNED-BAND SEQUENCE MOTIFS AND PWMs (exploratory)

Phase 5 was run on the earlier 97,198-band catalog and has not been repeated on
the current seed-stable catalog. Its motif models add essentially no held-out
discrimination beyond biophysical features. It is retained for provenance but
is not a primary manuscript analysis.

Scientific questions:

1. Does sequence around a positive or negative band differ from same-protein,
   same-Q8 non-band sequence?
2. Do train-discovered motifs replicate in validation and test proteins?
3. Does sequence improve held-out band-selection prediction beyond Phase 3A
   biophysical features?

Inputs:

- Earlier averaged-band catalog:
  results/publication_comparable_v2/
    analysis_seed_averaged_signed_bands/signed_bands.csv

- Residue sequence and Q8 annotations:
  results/publication_comparable_v2/
    analysis_seed_averaged_band_biophysics/annotations/
      residue_biophysical_annotations.csv.gz

- Phase 3A Q8-segment candidates:
  results/publication_comparable_v2/
    analysis_seed_averaged_band_phase3a/q8_segment_candidates.csv.gz

### 5A: APEX-CENTERED PWM AND LOCAL MOTIFS

Script:

```text
signed_band_analysis/analyze_signed_band_apex_pwm.py
```

Design:

- Extract a ±10-residue window around every band apex.
- Compare each apex with up to five same-protein, exact-Q8 control anchors.
- Require control windows to remain outside positive and negative band
  exclusion regions.
- Weight each band’s controls so their combined weight equals one.
- Generate pooled and Q8-stratified amino-acid PWMs.
- Discover exact amino-acid and reduced-alphabet 2–5-mers using train proteins.
- Lock train discoveries and evaluate them unchanged in validation and test.
- Use within-protein band-minus-control effects, exact protein sign tests and
  Benjamini-Hochberg correction.

Reduced alphabet:

- B, positive/basic: K, R, H
- N, negative/acidic: D, E
- P, polar uncharged: S, T, N, Q, C
- H, hydrophobic: A, V, I, L, M
- A, aromatic: F, W, Y
- T, turn-associated: G, P

Motifs are counted anywhere in the ±10 window and need not cross the apex.

Coverage:

- input bands: 97,198
- bands with an eligible matched background: 67,737
- match rate: 69.7%

Main PWM results:

Positive apices:

- M, I, C, W, F, L and V were depleted in all six test sets.
- S, D and G were enriched in all six.
- T was enriched in five of six.

Negative apices:

- G and P were depleted in all six test sets.
- I and L were enriched in all six.

Exact-motif replication was sparse:

- Positive-window GG replicated in validation and test for ESM3 frozen and
  ESM3 top4.
- No exact 2–5-mer replicated in both held-out splits for all six conditions.

Reduced motifs were more portable:

- 21 model/motif rows replicated in validation and test.
- Positive PTT replicated in four conditions.
- Positive TPT replicated in three conditions.
- No reduced motif replicated across all six conditions.

Interpretation:

Positive bands favor polar, turn-associated and hydrophobic-depleted sequence
context. Negative bands favor hydrophobic, helix-compatible context. The
signal is broader chemical composition rather than one universal exact motif.

Outputs:

```text
results/publication_comparable_v2/
  analysis_seed_averaged_band_apex_pwm/<condition>/
    pwm_amino_acid_frequencies.csv.gz
    pwm_coverage_summary.csv
    matched_apex_windows.csv.gz
    train_discovered_kmers.csv
    kmer_cross_split_replication.csv
    train_discovered_reduced_motifs.csv
    reduced_motif_cross_split_replication.csv
    reduced_alphabet_mapping.csv
    parameters.json
```

### 5B: COMPLETE Q8-SEGMENT MOTIF ANALYSIS

Script:

```text
signed_band_analysis/analyze_signed_band_sequence_motifs.py
```

Design:

- Treat the complete Q8 segment containing an apex as the selected object.
- Exclude segments containing both positive and negative apices.
- Match each case to up to five clean segments of the same Q8 type in the same
  protein.
- Clean controls contain no apex and overlap no signed band interval.
- Place the control anchor at the same relative position within its segment as
  the case apex.
- Test position-specific amino acids in ±3, ±5 and ±10 windows.
- Test exact 2–5-mers in the complete segment and in a band-width-matched
  interval.
- Discover features on train proteins and lock them at:

      BH q <= 0.05
      absolute log2 odds ratio >= 1

- Evaluate locked features without rediscovery on validation and test.

Coverage:

- matched selected-band sets: 82,367
- matched case/control rows: 364,737
- each condition retained the top 100 position features and 100 exact k-mers

Replicated patterns:

Negative Q8-H segments:

- alanine at offsets -5, -4 and +5 replicated in all six conditions
- helix-compatible dipeptides including RA, VR and AI replicated in all six

Positive segments:

- methionine depletion at Q8-C apices replicated in all six conditions
- isoleucine and leucine depletion replicated in five conditions
- leucine depletion at positive Q8-T apices replicated in five
- band-width-matched GG enrichment replicated in three conditions

Held-out model performance:

| Model | Positive AUROC | Negative AUROC |
|---|---:|---:|
| Q8 + biophysical features | 0.812 | 0.818 |
| Motif features only | 0.557 | 0.681 |
| Biophysical features + motifs | 0.812 | 0.808 |

Adding motifs changed test AUROC by approximately:

- positive: +0.0001
- negative: -0.0103

Interpretation:

Sequence motifs alone contain some band-selection information, especially for
negative Q8-H segments. However, motifs add essentially no positive predictive
information and reduce negative performance after Phase 3A biophysical
features are included. Most sequence information is therefore redundant with
Q8 subtype, flexibility, exposure, length, geometry and disorder.

Outputs:

```text
results/publication_comparable_v2/
  analysis_seed_averaged_band_sequence_motifs/<condition>/
    matched_sequence_windows.csv.gz
    position_specific_enrichment.csv
    kmer_enrichment_train.csv
    locked_motif_definitions.json
    motif_validation_test_results.csv
    motif_model_family_consistency.csv
    motif_incremental_model_performance.csv
    sequence_logos/
    parameters.json
```

### PHASE 5 CONCLUSION

Across all six conditions:

- positive bands favor turn/polar and hydrophobic-depleted sequence
- negative bands favor hydrophobic, helix-compatible sequence
- broad chemical patterns replicate better than exact short motifs
- negative Q8-H segments contain the strongest exact sequence associations
- motifs provide little information beyond the Phase 3A biophysical features

The data support reproducible signed-band sequence chemistry, but not a
universal independent sequence code for band selection.

Limitations:

- Only 69.7% of apex bands had eligible Phase 5A backgrounds.
- Motifs can occur anywhere in the tested window and need not cross the apex.
- The six conditions use the same proteins and are not independent datasets.
- Sequence enrichment is associative and does not establish that a motif
  causally controls band strength or protein mechanics.

## CURRENT OVERALL CONCLUSIONS

1. Signed influence bands are highly reproducible across seeds, and their
   locations are strongly nonrandom relative to protein-preserving
   circular-shift nulls.

2. Raw positive/negative differences in flexibility, exposure and local
   structure are strong localization and model-accuracy diagnostics, but are
   not by themselves evidence of a specialized biological mechanism.

3. Under strict same-protein matching and union-group inference, positive
   apices retain greater torsional change and MD strain, whereas negative
   apices retain lower torsional change and strain and lie deeper inside Q3
   regions. These are the primary Phase 2 biological results.

4. The primary Phase 2 family contains 48 predeclared two-sided tests. Broader
   annotation screens and raw circular-shift enrichments are exploratory.

5. Positive bands are enriched for Q8 loops, Neq peaks and high-strain
   environments, but have low exact recall. Bands should not be interpreted as
   general loop, Neq-peak or high-strain detectors.

6. Complete-Q8-segment selection is predictable from biophysical features, but
   this post hoc object is not what the contribution model directly selects;
   Phase 3A remains exploratory.

7. At the exact apex, positive bands favor weakly connected and loosely packed
   experimental structures. The corresponding negative-apex hub claim does
   not survive strict matching and union-group inference. Complete negative
   band intervals can still occupy more connected environments; apex and
   interval results must not be conflated.

8. Neither apex nor segment analyses support a generic contact-community,
   ECOD-boundary or allosteric-hinge interpretation.

9. Evidence-only and shifted-attention controls show that \(s_j\) supplies much
   of the broad candidate landscape, while its learned alignment with \(B_j\)
   selects and amplifies a smaller set of detectable locations. This effect is
   strongest for positive bands and is more modest than the uniform-attention
   comparison alone suggested.

10. A band’s contribution is not distributed uniformly across query residues.
    After controlling for source band, query Q8 and sequence distance, query
    biophysics strongly predicts high receivers. Membership in another
    model-derived band is the largest unique predictor, followed by Neq and
    sign-specific local biophysics.

11. Positive bands preferentially influence flexible and exposed queries,
    whereas negative bands preferentially influence rigid and buried queries.
    Much of this influence is long range in sequence. Direct structural and
    putative polar-contact associations exist, but add little receiver
    prediction beyond query biophysics; crystallographic-water paths add no
    consistent predictive signal.

12. The legacy motif analysis suggests broad sequence-composition differences,
    but exact motifs add essentially no held-out discrimination beyond
    biophysical features. Because it used the earlier band catalog, it is not a
    primary conclusion of the current pipeline.

13. Overall, the model learns a distributed decision pattern in which
    structurally and mechanically distinctive source regions provide signed
    evidence, attention selects and amplifies evidence-bearing locations, and
    query biophysics is strongly associated with which residues receive that
    evidence. This is a model mechanism, not proof of a physical allosteric
    pathway.

## CURRENT LIMITATIONS AND NEXT PHASES

- NetSurfP annotations are predicted structural annotations. Experimental PDB
  geometry and contact networks provide complementary evidence, but structural
  coverage is incomplete for some segments and is somewhat lower for positive
  loop segments.

- Fully resolved local, geometry-complete and strict contact-graph sensitivity
  analyses have been completed. They preserve the main directions, although
  some small incremental structural AUROC gains remain uncertain.

- Strain is available only for the 208 test proteins. Its associations have not
  yet been replicated using independent train or validation strain datasets,
  and strain was not used as a train-learned incremental predictor.

- Phase 3A and the secondary segment-level part of Phase 3B treat complete Q8
  segments as candidate objects. The primary Phase 3B analysis uses the actual
  band apex instead.

- The compact, sign-constrained band intervals should not be treated as
  independent physical domains. Catalogs from different model conditions can
  identify overlapping biological regions and should not be summed as unique
  sequence coverage.

- The locked \(R_p\geq2\) detector threshold was chosen before the downstream
  test-set biological analyses, but a formal multi-threshold sensitivity sweep
  has not been completed. The detector is scale-invariant and therefore does
  not measure absolute influence magnitude.

- ECOD boundaries are annotations rather than experimentally established hinge
  axes. Contact-network measurements also depend on structural completeness
  and the selected contact definition.

- Phase 4 includes pairwise 3D geometry, direct contacts, contact-network
  paths, communities, ECOD relationships, putative polar contacts and
  crystallographic-water paths. These remain static structural associations;
  strain covariance and dynamic solvent-mediated transmission were not tested.

- Only 69.7% of apex bands had an eligible same-protein, same-Q8 matched
  background in the apex-centered motif analysis.

- Motifs may occur anywhere inside their tested window or segment and are not
  necessarily centered on or crossing the band apex. Exact motif analysis also
  involves many correlated tests despite multiple-testing correction and
  held-out replication.

- The six model conditions are technical conditions applied to the same
  proteins, not six independent biological cohorts. Consistency across
  conditions demonstrates model robustness but does not constitute six
  independent biological replications.

- The analyses establish reproducible associations, held-out predictive
  relationships and exact internal model decompositions. They do not establish
  causal biochemical or mechanical transmission.

- Experimental perturbations, especially mutations designed to alter band
  evidence, packing, strain or receiver relationships, are needed to test
  causality.

- The remaining work is manuscript-level synthesis and figure selection, not a
  new mechanistic study.
