ESMfluc signed-contribution / I_j band analysis

REPOSITORY ROOT
/home/zahralab/Desktop/ESMfluc/scripts/final_pipeline

All relative paths below are relative to this directory.

========================================================================
SCIENTIFIC OBJECTIVE
========================================================================

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

========================================================================
CORE DEFINITIONS
========================================================================

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

========================================================================
ENSEMBLE RULE
========================================================================

Final bands were detected from the three-seed arithmetic average:

    mean_seed(I_j)
      = [I_j(seed1) + I_j(seed2) + I_j(seed3)] / 3

Per-seed bands were analyzed first to verify reproducibility. The final
biophysical analyses use bands detected from mean_seed(I_j).

========================================================================
DATASETS AND MODELS
========================================================================

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
- approximately 37.65 GB compressed

Maximum audited reconstruction error for:

    I_j = s_j * B_j

was 1.91e-6.

========================================================================
PRIMARY DATA LOCATIONS
========================================================================

Data splits and NetSurfP:

data_splits/atlas_grouped_v1/

All contribution-file manifest:

results/publication_comparable_v2/
  all_split_signed_contributions_manifest.tsv

Contribution audit:

results/publication_comparable_v2/
  all_split_signed_contributions_audit.json

Three-seed averaged-profile manifest:

results/publication_comparable_v2/
  seed_averaged_signed_contributions_manifest.tsv

Seed-average audit:

results/publication_comparable_v2/
  seed_averaged_influence_audit.tsv

Each run’s contribution files are under:

results/publication_comparable_v2/runs/<condition>/seed_<seed>/
  all_split_signed_contributions/

Each protein record contains:

- contribution_matrix:
  C_ij, shape LxL

- intrinsic_signed_evidence:
  s_j, shape L

- signed_column_influence:
  I_j, shape L

- attention_matrix:
  A_ij, shape LxL

- attention_column_mean:
  B_j, shape L

- seed_averaged_signed_column_influence:
  mean_seed(I_j), shape L

========================================================================
DATA-GENERATION SCRIPTS
========================================================================

Attention/extract_all_split_signed_contributions.py

Attention/audit_all_split_signed_contributions.py

signed_band_analysis/add_seed_averaged_influence.py

The extraction runner used both GPUs:

run_all_split_signed_contributions_v2.sh

========================================================================
PHASE 1: BAND DETECTION AND SEED REPRODUCIBILITY
========================================================================

Band detector:

signed_band_analysis/extract_signed_contribution_bands.py

------------------------------------------------------------------------
1A. Band-calling algorithm
------------------------------------------------------------------------

Positive and negative I_j profiles are analyzed separately.

Algorithm:

1. Smooth the signed profile using windows of 1, 3 and 5 residues.
2. Detect positive local maxima or negative local minima with
   scipy.signal.find_peaks.
3. Estimate a robust local signal scale from the median absolute deviation.
4. Require peak prominence to be at least 2.5 times the robust MAD-derived
   scale.
5. Require the peak to persist across at least two of the three smoothing
   scales.
6. Cluster peaks from different smoothing scales when their apices are within
   two residues.
7. Enforce a minimum same-scale peak distance of three residues.
8. Define band boundaries from median half-prominence limits.
9. Store the apex, sign, start, end, width, sequence, prominence and smoothing
   scale persistence.

Band strength therefore matters: the detector does not classify every local
positive or negative fluctuation as a band.

Band width is determined by the half-prominence boundaries of the detected
multi-scale peak. It is not a fixed window and should not be interpreted as a
physical domain boundary.

------------------------------------------------------------------------
1B. Per-seed bands
------------------------------------------------------------------------

Outputs:

results/publication_comparable_v2/analysis_signed_bands/
  signed_bands.csv
  signed_band_protein_summary.csv
  signed_band_parameters.json

There are 294,459 per-seed bands:

- seed 1: 98,011
- seed 2: 96,753
- seed 3: 99,695

------------------------------------------------------------------------
1C. Seed reproducibility
------------------------------------------------------------------------

Script:

signed_band_analysis/analyze_signed_band_seed_reproducibility.py

Outputs:

results/publication_comparable_v2/
  analysis_signed_band_reproducibility/
    seed_pair_reproducibility_by_protein.csv
    seed_pair_reproducibility_summary.csv
    seed_consensus_bands.csv
    consensus_reproducibility_summary.csv
    seed_pair_block_shift_null.csv
    consensus_block_shift_null.csv
    seed_reproducibility_parameters.json

Logic:

- Match same-sign bands between seed pairs using one-to-one positional
  matching.
- Compare observed overlap with protein-level circular block-shift nulls that
  preserve band counts and relative patterns.
- Construct seed-consensus bands supported by at least two seeds.

Results across condition/split/sign cells:

- pairwise micro-Jaccard: 0.620–0.883; median 0.726
- pairwise micro-F1: 0.765–0.938; median 0.842
- mean matched-apex separation: 0.22–1.58 residues
- Jaccard-null z-scores: 35.4–133.6
- consensus counts: 1.13–1.35 times block-shift expectation
- all reported consensus tests were strongly nonrandom

Conclusion:

Band locations are reproducible across seeds and cannot be explained by the
number and spacing of randomly shifted bands.

------------------------------------------------------------------------
1D. Final seed-averaged bands
------------------------------------------------------------------------

The same detector was applied to mean_seed(I_j).

Outputs:

results/publication_comparable_v2/
  analysis_seed_averaged_signed_bands/
    signed_bands.csv
    signed_band_protein_summary.csv
    signed_band_parameters.json

Final counts:

- 97,198 total bands
- 50,257 positive bands
- 46,941 negative bands
- 5.075 total bands per 100 residues
- positive density: 2.624 per 100 residues
- negative density: 2.451 per 100 residues

Widths:

- positive mean width: 11.59 aa
- positive median width: 9 aa
- negative mean width: 18.42 aa
- negative median width: 14 aa

Apex distances between consecutive bands in the complete ordered band list:

- +/+ mean 16.28, median 14 residues
- -/- mean 22.53, median 17 residues
- opposite-sign mean 17.58, median 13 residues

Same-sign distance while ignoring intervening opposite-sign bands:

- + to next +: mean 34.78, median 30 residues
- - to next -: mean 34.96, median 30 residues

Approximately 71% of consecutive opposite-sign intervals overlap or touch.

Consequences:

- The reported band density is an apex density, not nonoverlapping sequence
  coverage.
- Band widths must not be summed to estimate the percentage of the protein
  covered by independent bands.
- Positive and negative bands may overlap because they were detected from
  positive and negative profiles separately.

========================================================================
PHASE 2: BIOPHYSICAL ENRICHMENT, NONRANDOMNESS AND IDENTIFIER ANALYSIS
========================================================================

Phase 2 consists of all analyses implemented in:

signed_band_analysis/analyze_signed_band_biophysical_enrichment.py

The annotation preparation is performed by:

signed_band_analysis/annotate_signed_bands_with_netsurfp.py

The phase has four main questions:

2A. Where are positive and negative band apices located biophysically?

2B. Are those locations nonrandom relative to a protein-preserving positional
    null?

2C. What fraction of biological annotations is captured by bands—the inverse
    probability P(band detects annotation)?

2D. After comparing apices with same-protein, same-Q3 non-band residues, what
    properties still distinguish band apices?

A test-only strain extension was subsequently added to the same enrichment
script.

------------------------------------------------------------------------
2A. Residue and band annotation
------------------------------------------------------------------------

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

results/publication_comparable_v2/
  analysis_seed_averaged_band_biophysics/annotations/
    residue_biophysical_annotations.csv.gz
    signed_bands_biophysical_annotations.csv.gz
    annotation_audit.json

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

------------------------------------------------------------------------
2B. Raw apex localization and circular-shift nonrandomness
------------------------------------------------------------------------

For each protein, condition and sign, the script circularly shifts the complete
same-sign apex pattern within the eligible protein interval.

The null preserves:

- protein identity
- model condition
- sign
- number of bands
- relative band spacing
- clustering pattern

It changes only the absolute positions of the apex pattern.

The analysis uses:

- 1,000 circular block shifts
- equal protein weighting
- protein-level observed means
- empirical upper, lower and two-sided p-values
- Benjamini-Hochberg correction

Representative test-set results across the six model conditions:

Property at apex              Positive       Negative       Shifted null
---------------------------------------------------------------------------
Neq                           2.03–2.09      1.02–1.03      1.36–1.37
Q3 coil                       93.3–94.6%     1.6–3.2%       43.4–43.9%
Q8 C/T/S                      91.1–92.9%     1.2–3.1%       41.2–41.6%
Structured linker/loop        66.9–77.7%     0.6–1.9%       ~29%
RSA                           0.52–0.55      0.17–0.24      ~0.34
RSA >= 0.25                   88.8–92.0%     26.7–41.8%     ~59%
Distance to Neq peak          1.8–2.6 aa     7.8–8.4 aa     ~5.2 aa
Torsional change              90–97 degrees  7–9 degrees    ~47 degrees

Interpretation:

- Positive apices preferentially occupy flexible, exposed, loop-like,
  torsionally changing environments near Neq peaks.
- Negative apices preferentially occupy rigid, structured, less exposed
  environments with small local torsional changes.
- The apex locations are strongly nonrandom.
- These raw differences do not establish a specialized control mechanism
  because they partly reflect the flexible/rigid state associated with the
  signed contribution.

The script also performs paired positive-versus-negative protein-level
contrasts. These are stored in:

paired_flex_vs_rigid_summary.csv

------------------------------------------------------------------------
2C. Inverse coverage and identifier analysis
------------------------------------------------------------------------

The raw apex analysis estimates:

    P(annotation | band)

For example:

    What percentage of positive apices are Q8 C/T/S?

The inverse analysis estimates:

    P(band detects annotation)

For example:

    What percentage of all Q8 C/T/S residues or segments are detected by a
    positive band?

Proteins with zero bands are retained in the denominators.

Detection methods include:

- exact apex
- within one residue of an apex
- within two residues
- within five residues
- overlap with a full band interval
- segment contains an apex
- segment overlaps a band interval

Outputs:

results/publication_comparable_v2/
  analysis_seed_averaged_band_biophysics/enrichment/
    annotation_band_coverage_by_protein.csv.gz
    annotation_band_coverage_identifier_summary.csv

Positive-band test results for Q8 C/T/S:

- exact-apex precision: 91.1–92.9%
- exact-apex coverage of all C/T/S residues: 5.96–7.05%
- band-interval precision: 65.3–68.0%
- band-interval coverage of C/T/S residues: 41.9–52.6%
- C/T/S segments containing a positive apex: 31.1–36.7%
- C/T/S segments overlapping a positive band: 41.3–50.7%

Positive-band test results for Neq peaks:

- exact-apex precision: 18.9–25.5%
- exact-apex coverage of Neq peaks: 8.2–9.2%
- Neq-peak coverage within two residues: 29.7–35.5%
- Neq-peak coverage by positive band intervals: 43.2–52.5%

Conclusion:

- A positive apex is usually loop-like.
- Most loop-like residues and segments do not contain an apex.
- Positive bands are not general Q8-loop or Neq-peak detectors.
- This asymmetry motivated Phase 3A: why is one plausible Q8 segment selected
  while another same-Q8 segment in the same protein is not?

------------------------------------------------------------------------
2D. Same-protein, within-Q3 matched analysis
------------------------------------------------------------------------

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

Control reuse is allowed across different apex match sets but not within one
set.

The q3_only results are stored compactly as one case row containing the number
and means of all eligible controls. Millions of redundant individual-control
rows are not stored.

Q3-only coverage:

- total apices: 97,198
- matched apices: 95,793
- match rate: 98.55%
- implied control assignments: 3,735,187
- mean controls per matched apex: approximately 39
- every control comes from the same protein

Inference:

- calculate case-minus-control effects within each protein
- give proteins equal weight
- use protein-level sign flips
- use protein bootstrap confidence intervals
- apply Benjamini-Hochberg correction

Dominant test-set findings:

Positive Q3-C apices versus same-protein Q3-C non-band controls:

- Neq: +0.42 to +0.56
- RSA: +0.092 to +0.118
- torsional change: +17.2 to +27.2 degrees
- Q8-S enrichment: +5.8 to +7.3 percentage points
- structured-linker enrichment: +11.5 to +30.5 points
- Neq-peak enrichment: +6.1 to +12.8 points

Negative Q3-H apices versus same-protein Q3-H controls:

- Neq: -0.17 to -0.23
- RSA: -0.047 to -0.126
- torsional change: -21.9 to -24.9 degrees
- Q8-H enrichment: +10.8 to +13.0 percentage points
- 57.0–63.4 percentage points less likely to be near a Q3 boundary

Negative Q3-E apices versus same-protein Q3-E controls:

- Neq: -0.103 to -0.129
- RSA: -0.083 to -0.107
- torsional change: -16.3 to -18.9 degrees
- Q8-E enrichment: +1.3 to +2.5 percentage points
- 16–29 percentage points less likely to be near a boundary

Normalized sequence position was weak and inconsistent.

Interpretation:

Q3 alone does not explain selection. Within the same protein and Q3 class:

- positive coil apices are more flexible, exposed and torsionally active than
  non-band coils
- negative helical and strand apices are more rigid, buried and internally
  positioned within structured segments than matched non-band residues

The stricter schemes ask conditional questions such as whether Q8 subtype,
torsion or boundary geometry remains different after additionally holding Neq,
RSA and position approximately constant.

------------------------------------------------------------------------
2E. Test-set strain extension
------------------------------------------------------------------------

Strain source:

/home/zahralab/MDStrainMapper/results/atlas_grouped_v1_test

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

results/publication_comparable_v2/
  analysis_seed_averaged_band_biophysics/
    enrichment_with_test_strain/

Raw circular-shift results across the six model conditions:

Positive apices:

- mean strain: 0.101–0.110 versus 0.078–0.081 under the null
- strain gradient: 0.0271–0.0293 versus 0.0196–0.0201
- top-decile strain frequency: 22.5–27.7% versus 10.3%
- distance to top-decile strain: 8.83–9.91 versus 11.08–11.16 residues

Negative apices:

- mean strain: 0.0624–0.0657 versus 0.0803–0.0808 under the null
- strain gradient: 0.0140–0.0152 versus approximately 0.0201
- top-decile strain frequency: 1.58–3.10% versus approximately 10.3%
- distance to top-decile strain: 12.51–13.04 versus 11.03–11.11 residues

For these strain metrics, the circular-shift tests were significant across all
six conditions and both signs at the available permutation resolution:

    BH-adjusted two-sided q = 0.001092

Same-protein Q3-only strain results:

Positive Q3-C apices versus Q3-C controls:

- mean strain difference: +0.0167 to +0.0251
- strain-gradient difference: +0.00638 to +0.00770
- top-decile strain enrichment: +9.53 to +15.70 percentage points
- all three effects were significant in all six conditions

Negative Q3-H apices versus Q3-H controls:

- mean strain difference: -0.0190 to -0.0120
- strain-gradient difference: -0.00626 to -0.00344
- top-decile strain difference: -8.60 to -5.19 percentage points
- all three effects were significant in all six conditions

Negative Q3-E apices versus Q3-E controls:

- mean strain difference: -0.0137 to -0.00851
- strain-gradient difference: -0.00694 to -0.00358
- top-decile strain difference: -3.86 to -2.50 percentage points
- all three effects were significant in all six conditions

Distance to a high-strain residue was less robust after Q3-only control. The
local strain magnitude and strain gradient were the more consistent
within-Q3 signals.

High-strain identifier results for positive bands:

- exact-apex precision: 23.3–29.9%
- exact-apex coverage of high-strain residues: 5.96–7.66%
- band-interval precision: 15.9–18.3%
- band-interval coverage of high-strain residues: 51.6–56.1%

Conclusion:

Positive apices are enriched for locally high and rapidly changing strain,
even relative to same-protein Q3-C controls. Negative structured apices show
the opposite pattern.

However, most high-strain residues are not exact positive apices. Positive
bands are enriched markers of high-strain environments, not general
high-strain-residue detectors.

Because strain is currently test-only, these strain associations have not yet
been replicated on independent train/validation strain datasets.

------------------------------------------------------------------------
2F. Outputs, visualization and audit
------------------------------------------------------------------------

Primary non-strain enrichment outputs:

results/publication_comparable_v2/
  analysis_seed_averaged_band_biophysics/enrichment/
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
    biophysical_enrichment_parameters.json

Strain-aware outputs:

results/publication_comparable_v2/
  analysis_seed_averaged_band_biophysics/
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
      strain_input_audit.csv
      biophysical_enrichment_parameters.json

Visualization script:

signed_band_analysis/plot_signed_band_phase2_results.py

Existing figures:

results/publication_comparable_v2/
  analysis_seed_averaged_band_biophysics/phase2_figures/

These include:

- Q3 and Q8 composition plots
- continuous Neq, RSA and torsion compositions
- Q3-only matched effect plots
- inverse selection/identifier plots
- matching-control availability diagnostics
- stricter matching diagnostics

The currently generated figure directory predates the strain extension.
Strain results are currently available in the strain-aware tabular outputs but
have not yet been added to the existing Phase 2 figures.

Pipeline scripts:

signed_band_analysis/run_signed_band_biophysical_pipeline.sh

signed_band_analysis/audit_signed_band_biophysical_pipeline.py

Original non-strain audit:

results/publication_comparable_v2/
  analysis_seed_averaged_band_biophysics/pipeline_audit.json

- passed
- 42 checks
- 0 failures

Final strain-aware audit:

results/publication_comparable_v2/
  analysis_seed_averaged_band_biophysics/
    pipeline_audit_with_test_strain.json

- passed
- 45 checks
- 0 failures
- all 97,198 bands preserved
- all 319,199 residue annotations preserved
- all 208 test strain files valid
- controls confirmed outside all band intervals
- Q3 matches and matching calipers independently verified
- summary statistics independently recomputed

========================================================================
PHASE 3A: Q8-SEGMENT OBJECT SELECTION
========================================================================

Script:

signed_band_analysis/analyze_signed_band_object_selection.py

Scientific question:

Why is one plausible Q8 structural segment selected by the model as an
influential band object while another segment of the same Q8 subtype in the
same protein is not?

This phase changes the unit of analysis from an individual residue to a
complete contiguous Q8 segment.

------------------------------------------------------------------------
3A.1 Candidate, case and control definitions
------------------------------------------------------------------------

Candidate object:

- one complete contiguous Q8 segment

Positive case:

- segment contains one or more positive apices
- segment contains no negative apex

Negative case:

- segment contains one or more negative apices
- segment contains no positive apex

Clean control:

- same protein
- identical Q8 subtype
- contains no apex
- does not overlap any positive or negative band interval

Excluded objects:

- segments containing both positive and negative apices
- segments overlapping a band interval but containing no apex

The exclusion of overlap-without-apex segments prevents broad band shoulders
from being treated as clean non-band controls.

Neq, RSA, segment length and position are not used to select controls. They
remain candidate explanations for selection.

------------------------------------------------------------------------
3A.2 Object statistics
------------------------------------------------------------------------

There are 481,842 condition/split/Q8-segment rows.

Mutually exclusive categories:

- positive-only selected segments: 48,900
- negative-only selected segments: 45,513
- dual-sign segments: 1,208
- clean controls: 141,482
- overlap-without-apex excluded segments: 244,739

The large overlap-without-apex category reflects the broad, frequently
overlapping band intervals. Phase 3A therefore tests a specific apex-containing
Q8-object definition; it does not claim that every band interval aligns
perfectly with one Q8 segment.

------------------------------------------------------------------------
3A.3 Feature groups
------------------------------------------------------------------------

Sequential feature stages:

1. Q8 only

2. Add Neq:
   - mean Neq
   - maximum Neq
   - Neq-peak fraction
   - Neq-peak excess

3. Add exposure, length and position:
   - mean RSA
   - maximum RSA
   - log segment length
   - normalized midpoint

4. Add geometry and boundaries:
   - mean torsional change
   - maximum torsional change
   - Q3-boundary fraction
   - mean distance to Q3 boundary
   - structured-linker fraction

5. Add disorder:
   - mean disorder
   - maximum disorder

6. Add coarse sequence composition:
   - glycine fraction
   - proline fraction
   - hydrophobic fraction
   - charged fraction
   - aromatic fraction
   - sequence entropy

------------------------------------------------------------------------
3A.4 Modeling and held-out evaluation
------------------------------------------------------------------------

Models are trained only on the train split.

They are evaluated without refitting on:

- validation proteins
- test proteins

Weights balance selected cases and clean controls within protein/Q8 strata.

Primary held-out metric:

- weighted AUROC

Additional metrics include:

- weighted average precision
- within-protein/Q8 concordance

Mean test AUROC across six model conditions:

Stage                                  Positive    Negative
----------------------------------------------------------------
Q8 only                               0.500       0.500
Add Neq                               0.712       0.663
Add RSA/length/position               0.760       0.803
Add geometry/boundaries               0.804       0.803
Add disorder                          0.812       0.818
Add sequence composition              0.812       0.821

Final validation AUROC was similarly strong:

- positive: approximately 0.811
- negative: approximately 0.821

Therefore, the feature-based discrimination generalized from train proteins to
both held-out validation and test proteins.

------------------------------------------------------------------------
3A.5 Biological interpretation
------------------------------------------------------------------------

Positive selection:

Selected positive Q8-S/T segments generally have:

- higher Neq
- higher RSA
- greater segment length
- greater torsional or geometric transition signal
- slightly higher disorder
- fewer internal Q3 boundaries

Negative selection:

Selected negative Q8-H/E segments generally have:

- lower Neq
- fewer Neq peaks
- lower mean torsional change
- lower disorder
- greater length
- fewer internal Q3 boundaries

Feature-stage conclusions:

- Neq is important but insufficient.
- RSA, length and position add substantial held-out information.
- Geometry and boundary features add particularly strong information for
  positive selection.
- Disorder adds a smaller but reproducible increment.
- Coarse amino-acid composition adds almost no positive-band discrimination
  after the biophysical features and only a very small negative-band increment.
- This does not rule out specific local sequence motifs, which require a formal
  motif analysis.

Statistical safeguard:

- rows supported by fewer than 10 proteins remain descriptive
- those rows receive no confidence interval, p-value or q-value

------------------------------------------------------------------------
3A.6 Outputs
------------------------------------------------------------------------

results/publication_comparable_v2/
  analysis_seed_averaged_band_phase3a/
    q8_segment_candidates.csv.gz
    matched_feature_effects_by_protein.csv.gz
    matched_feature_effect_summary.csv
    sequential_model_performance.csv
    sequential_model_coefficients.csv
    phase3a_parameters.json

The output directory retains the historical “phase3a” name even though the
script itself was renamed to:

signed_band_analysis/analyze_signed_band_object_selection.py

========================================================================
PHASE 3B: EXTERNAL STRUCTURAL AND MECHANICAL EXPLANATION
========================================================================

Scientific questions:

1. Do band-selected Q8 segments occupy distinctive experimentally observed
   geometries or contact-network positions?

2. Do positive and negative bands distinguish mechanically deformable regions
   from structurally stabilizing regions?

3. Do experimental structure and domain features improve held-out prediction
   of band selection beyond the Phase 3A biophysical features?

4. Are the Phase 3C internal mechanism classes associated with different
   external structural or mechanical environments?

Models were fitted separately for each model condition and band sign.

Sequential structural models were:

- trained only on train proteins
- evaluated without refitting on validation and test proteins
- kept separate for positive and negative segment selection

Test-only strain was excluded from model training and held-out model-performance
claims.

------------------------------------------------------------------------
3B.1 Experimental structure acquisition and sequence mapping
------------------------------------------------------------------------

Contact-map builder:

Attention/build_contact_maps_from_pdb.py

Purpose:

- download and cache experimental PDB structures
- align observed PDB residues explicitly to model-sequence indices
- never assume that model-sequence position equals PDB residue number
- preserve PDB chain identifiers, residue numbers and insertion codes
- identify unresolved or missing residues
- record sequence-mapping identity and coverage
- record experimental method and resolution
- generate model-indexed C-alpha contact edges and distances
- optionally retain model-indexed C-alpha coordinates

Required Phase 3B options:

    --map_representation edges
    --include_ca_coordinates

Contact-map outputs:

results/publication_comparable_v2/
  analysis_seed_averaged_band_external_structure/contact_networks/

------------------------------------------------------------------------
3B.2 Structure coverage and mapping integrity
------------------------------------------------------------------------

Structure coverage:

- 1,382 of 1,383 structures loaded successfully
- 1,379 mappings passed the identity and coverage thresholds
- all validation and test structures were accepted
- three train structures had insufficient mapping coverage
- 4v4e_M failed because the legacy PDB download returned HTTP 404

The explicit sequence-to-structure mapping prevents PDB numbering, insertion
codes or unresolved residues from being mistaken for model-sequence indices.

------------------------------------------------------------------------
3B.3 Segment-level case and control definitions
------------------------------------------------------------------------

Analysis script:

signed_band_analysis/analyze_signed_band_external_structure.py

Candidate object:

- one complete contiguous Phase 3A Q8 segment

Positive case:

- segment contains a positive band apex
- segment contains no negative apex

Negative case:

- segment contains a negative band apex
- segment contains no positive apex

Clean control:

- same protein
- identical Q8 subtype
- contains no positive or negative apex
- does not overlap any positive or negative band interval

These definitions preserve the Phase 3A object-selection estimand while adding
experimental structural and mechanical features.

------------------------------------------------------------------------
3B.4 Experimental structural features
------------------------------------------------------------------------

Local experimental geometry:

- C-alpha curvature
- C-alpha virtual torsion
- segment end-to-end distance divided by backbone path length

Contact-network organization:

- contact degree
- inverse-distance-weighted contact degree
- betweenness centrality
- closeness centrality
- contact-community membership
- participation coefficient
- contact-community-boundary status

Domain organization:

- distance to the nearest ECOD-domain boundary
- cross-domain contact counts
- position inside or near annotated ECOD domains

Test-only mechanical strain:

- segment mean strain
- segment maximum strain
- strain variability
- spatial strain gradient

Internal/external integration:

- associations between experimental features and Phase 3C mechanism classes
- band-level external features joined to evidence-dominated,
  consultation-dominated, combined and other mechanism classes

------------------------------------------------------------------------
3B.5 Test-only strain analysis
------------------------------------------------------------------------

Strain source:

/home/zahralab/MDStrainMapper/results/atlas_grouped_v1_test

Scope:

- strain is available only for test proteins
- strain is analyzed through prespecified within-protein, same-Q8 matched
  effects
- strain is excluded from model training
- strain is excluded from train-to-validation/test incremental-performance
  claims

The strain analysis therefore tests whether selected test-set segments occupy
different mechanical environments than same-protein, same-Q8 clean controls. It
does not test whether strain is a train-learned predictor that generalizes to
new proteins.

------------------------------------------------------------------------
3B.6 Sequential held-out modeling
------------------------------------------------------------------------

Feature stages:

1. Q8 only

2. Add base biophysics

3. Add experimental geometry

4. Add contact-network features

5. Add ECOD-domain features

Mean test AUROC across the six model conditions:

Stage                                  Positive    Negative
----------------------------------------------------------------
Q8 only                               0.500       0.500
Base biophysics                       0.767       0.798
Add experimental geometry             0.806       0.800
Add contact network                   0.815       0.805
Add ECOD/domain features              0.831       0.822

Interpretation:

- Experimental geometry substantially improves positive-band selection beyond
  the base biophysical features.
- Contact-network information adds further positive and smaller negative
  predictive information.
- ECOD/domain organization provides the strongest final held-out performance
  for both signs.
- Final test AUROC reaches approximately 0.83 for positive selection and 0.82
  for negative selection.

------------------------------------------------------------------------
3B.7 Main biological results
------------------------------------------------------------------------

Positive selected segments tend to have:

- higher mechanical strain
- weaker contact-network connectivity
- Q8 C/S/T loop-like environments
- signatures consistent with mechanically deformable or weakly packed regions

Negative selected segments tend to have:

- lower mean mechanical strain
- greater contact density
- greater contact-network centrality
- Q8 H/E structured environments
- signatures consistent with stabilizing contact-network cores or hubs

Boundary results:

- positive bands are not enriched at contact-community boundaries
- positive bands are not enriched at ECOD-domain boundaries
- selected segments of both signs generally occur farther inside annotated
  ECOD domains

Therefore, the positive-band signal is better described as local mechanical
deformability or weak packing than as generic domain-boundary or hinge
localization.

------------------------------------------------------------------------
3B.8 Integration with Phase 3C mechanisms
------------------------------------------------------------------------

Positive combined bands show the strongest combination of:

- high strain
- low connectivity
- mechanically deformable local structure

This is consistent with their Phase 3C mechanism:

- strong intrinsic flexibility-supporting evidence
- combined with broad consultation by other residues

Negative evidence-dominated Q8-E bands show a clear packed-core signature:

- low strain
- dense contact organization
- greater network centrality

This is consistent with strong intrinsic rigidity-supporting evidence arising
from a structurally stabilizing environment.

These results connect the model’s internal decomposition to experimentally
derived external structure, but remain associative rather than causal.

------------------------------------------------------------------------
3B.9 Outputs
------------------------------------------------------------------------

Result directory:

results/publication_comparable_v2/
  analysis_seed_averaged_band_external_structure/

Important outputs:

- structure_mapping_audit.csv
- external_features_by_q8_segment.csv.gz
- matched_external_effects_by_protein.csv.gz
- matched_external_effect_summary.csv
- sequential_external_model_performance.csv
- mechanism_class_external_associations.csv
- mechanism_band_external_features.csv.gz
- external_feature_coverage_summary.csv
- parameters.json

------------------------------------------------------------------------
3B.10 Tests and integrity checks
------------------------------------------------------------------------

Tests:

tests/test_build_contact_maps_from_pdb.py

tests/test_analyze_signed_band_external_structure.py

Status:

- all eight tests pass

Integrity protections include:

- explicit model-sequence-to-PDB mapping
- mapping identity and coverage thresholds
- preservation of missing-residue information
- separation of train fitting from validation/test evaluation
- test-only isolation of strain
- same-protein, identical-Q8 controls
- exclusion of controls overlapping any signed band

------------------------------------------------------------------------
3B.11 Limitations
------------------------------------------------------------------------

- The six model conditions use the same biological proteins and are not six
  independent biological datasets.

- Strain is test-only and was not evaluated as a train-learned incremental
  predictor.

- Positive loop segments have somewhat more incomplete experimental structural
  coverage.

- A sensitivity analysis restricted to fully resolved segments is still
  recommended.

- ECOD boundaries are annotations and should not be interpreted as
  experimentally established hinge axes.

- Contact-network centrality depends on the selected contact definition and
  available experimental coordinates.

- The results support associations between band selection and external
  mechanical or structural environments; they do not by themselves establish
  causal mechanical control.
========================================================================
PHASE 3C: INTERNAL EVIDENCE/CONSULTATION MECHANISM
========================================================================

Script:

signed_band_analysis/analyze_signed_band_model_mechanism.py

Scientific questions:

1. Is a band influential because the key has unusually strong intrinsic signed
   evidence s_j?

2. Is it influential because attention consults that key unusually broadly,
   through B_j?

3. Are both mechanisms elevated?

4. Do positive and negative bands use these mechanisms differently?

------------------------------------------------------------------------
3C.1 Matched control design
------------------------------------------------------------------------

For each band apex, controls are:

- in the same protein
- in the eligible protein interval
- exact same Q8 subtype as the apex
- outside every positive and negative band interval

The control value is the mean over all eligible same-protein, same-Q8 control
residues.

This is a residue-level internal-mechanism analysis, distinct from the
Q8-segment object analysis in Phase 3A.

------------------------------------------------------------------------
3C.2 Seed-aware exact decomposition
------------------------------------------------------------------------

Bands were called from mean_seed(I_j), but the identity:

    I_j = s_j * B_j

holds exactly within each individual seed.

Therefore, for each seed separately:

    delta log|I|
      = delta log|s|
      + delta log(B)

The script:

1. Extracts case and control values separately for each seed.
2. Computes the exact within-seed evidence and consultation differences.
3. Verifies the log decomposition.
4. Only then averages the effects across the three seeds.

This avoids incorrectly multiplying seed-averaged s_j and seed-averaged B_j.

------------------------------------------------------------------------
3C.3 Mechanism classes
------------------------------------------------------------------------

For a band with positive total log-magnitude enrichment:

- evidence-dominated:
  evidence contributes at least two-thirds of the positive enrichment, or
  evidence increases while consultation does not

- consultation-dominated:
  consultation contributes at least two-thirds, or consultation increases
  while evidence does not

- combined:
  both increase and neither contributes more than two-thirds

Additional classes:

- not_magnitude_enriched:
  total matched log|I| enrichment is zero or negative

- unclassified:
  no eligible exact-Q8 control or the decomposition cannot support a class

------------------------------------------------------------------------
3C.4 Coverage and seed stability
------------------------------------------------------------------------

Total bands:

- 97,198

Bands with eligible exact-Q8 controls:

- 92,806

Bands without eligible controls:

- 4,392
- these remain unclassified
- they do not enter matched case/control summaries

Seed-direction stability:

- 96,365 of 97,198 bands had the same direction in all three seeds
- 99.14% complete three-seed direction agreement
- 740 bands agreed in two of three seeds
- 93 bands agreed in one of three seeds

------------------------------------------------------------------------
3C.5 Positive-band mechanism results
------------------------------------------------------------------------

Median test-set effects across the six conditions:

- delta log|s|: 1.19
- intrinsic evidence enrichment: approximately 3.3-fold

- delta log(B): 1.03
- consultation enrichment: approximately 2.8-fold

- delta log|I|: 2.23
- total influence enrichment: approximately 9.3-fold

Average test-set mechanism fractions across conditions:

- combined: 77.9%
- evidence-dominated: 14.4%
- consultation-dominated: 1.6%
- not magnitude-enriched: 1.6%
- unclassified: 4.5%

Interpretation:

Positive bands usually become strong because they combine:

- strong intrinsic flexibility-supporting evidence
- unusually broad consultation by other residues

Neither component alone explains most positive bands.

------------------------------------------------------------------------
3C.6 Negative-band mechanism results
------------------------------------------------------------------------

Median test-set effects across the six conditions:

- intrinsic evidence enrichment: approximately 3.2-fold
- consultation enrichment: approximately 1.5-fold
- total |I| enrichment: approximately 5.1-fold

Average test-set mechanism fractions:

- evidence-dominated: 56.6%
- combined: 35.9%
- consultation-dominated: approximately 0.09%
- not magnitude-enriched: 4.2%
- unclassified: 3.3%

Interpretation:

Negative bands are more commonly evidence-dominated. Attention breadth still
amplifies negative evidence, but less strongly than for positive bands.

Purely consultation-driven negative bands are exceptionally rare.

------------------------------------------------------------------------
3C.7 Audited numerical identities
------------------------------------------------------------------------

- mean of the three seed I_j values reproduces the stored averaged band I_j
  with maximum error 1.11e-16

- maximum log-decomposition reconstruction error: 3.03e-7

- all 54 condition/seed/split cache shards were present

- all 291,594 expected band/seed rows were present:

      97,198 bands x 3 seeds

Conclusion:

Positive and negative bands use systematically different internal mechanisms:

- positive bands are usually combined evidence-plus-consultation objects
- negative bands are more often strong intrinsic evidence objects with a
  smaller attention-breadth amplifier

------------------------------------------------------------------------
3C.8 Outputs
------------------------------------------------------------------------

results/publication_comparable_v2/
  analysis_seed_averaged_band_phase3c/
    mechanism_by_band_and_seed.csv.gz
    mechanism_by_band_seed_averaged.csv.gz
    mechanism_effects_by_protein.csv.gz
    mechanism_effect_summary.csv
    mechanism_class_summary.csv
    per_seed_split_cache/
    phase3c_parameters.json

The output directory retains the historical “phase3c” name even though the
script itself was renamed to:

signed_band_analysis/analyze_signed_band_model_mechanism.py


========================================================================
PHASE 4: QUERY RECEIVERS OF SIGNED BAND CONTRIBUTIONS
========================================================================

Scientific question:

For one fixed influential source band, why does one query residue receive a
large contribution while another query in the same protein receives little?

For query residue i and source band b:

    R_i,b = sign_b * sum_(j in band b) C_ij

where larger R_i,b means that query i receives a stronger contribution in the
direction represented by the source band.

For one fixed key:

    C_ij = s_j * A_ij

Because s_j is constant across queries, query-to-query variation for that key
comes from attention A_ij. For a multi-residue band, the receiver profile sums
the contributions from all keys in the band.

------------------------------------------------------------------------
4.1 Script and inputs
------------------------------------------------------------------------

Script:

signed_band_analysis/analyze_signed_band_query_receivers.py

Inputs:

- Contribution manifest:
  results/publication_comparable_v2/
    all_split_signed_contributions_manifest.tsv

- Final averaged bands and eligible intervals:
  results/publication_comparable_v2/
    analysis_seed_averaged_signed_bands/
      signed_bands.csv
      signed_band_protein_summary.csv

- Residue annotations:
  results/publication_comparable_v2/
    analysis_seed_averaged_band_biophysics/annotations/
      residue_biophysical_annotations.csv.gz

- Phase 3C mechanism classes:
  results/publication_comparable_v2/
    analysis_seed_averaged_band_phase3c/
      mechanism_by_band_seed_averaged.csv.gz

Query features:

- Q8 subtype
- Neq
- RSA
- disorder
- torsional change
- coarse amino-acid class
- normalized sequence position
- whether the query lies inside any signed band

------------------------------------------------------------------------
4.2 Receiver profiles and matching
------------------------------------------------------------------------

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

------------------------------------------------------------------------
4.3 Data coverage
------------------------------------------------------------------------

All six model conditions were analyzed separately.

Total band-query rows:

- ESM2 frozen: 5,861,284
- ESM2 top4: 5,415,005
- ESM2 top28: 5,155,195
- ESM3 frozen: 5,628,953
- ESM3 top4: 5,427,178
- ESM3 top28: 5,660,699
- total: 33,148,314

For predictive modeling, high and low receivers are balanced within each
band/Q8/distance stratum, with at most 50 rows per receiver class and protein.

------------------------------------------------------------------------
4.4 Held-out receiver models
------------------------------------------------------------------------

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

3. Add source-band Phase 3C mechanism class.

Mean test AUROC across six conditions:

Stage                              Negative source   Positive source
--------------------------------------------------------------------
Distance/Q8 baseline                   0.507             0.514
Add query features                     0.883             0.810
Add source mechanism                   0.883             0.811

Validation performance was similar:

- negative mean AUROC: 0.882
- positive mean AUROC: 0.816

The baseline is near chance because high and low examples were deliberately
balanced within Q8 and distance strata. The large improvement therefore comes
from query biophysics rather than simple sequence proximity or Q8 identity.

Adding source mechanism class changed mean test AUROC by only:

- negative: +0.00007
- positive: +0.00038

The mechanism-class main effect therefore contributes almost no additional
receiver discrimination once query properties are known. Mechanism-by-query
interactions were not tested.

------------------------------------------------------------------------
4.5 Feature importance
------------------------------------------------------------------------

Mean test AUROC changes:

Feature                    Negative: single / unique   Positive: single / unique
-------------------------------------------------------------------------------
Neq                               +0.296 / +0.061             +0.252 / +0.061
RSA                               +0.165 / +0.008             +0.130 / +0.028
Torsional change                  +0.324 / +0.037             +0.069 / +0.003
Query lies in any band            +0.101 / +0.008             +0.137 / +0.015
Disorder                          +0.153 / +0.001             -0.005 / +0.011
Amino-acid class                  +0.100 / +0.004             +0.078 / +0.001
Normalized position              approximately zero          approximately zero

“Single” is the gain from adding one feature to the baseline.

“Unique” is the loss from removing that feature from the full query model.

Interpretation:

- Neq is the largest unique predictor for both signs.
- Torsional change is the second-largest unique predictor for negative
  receivers.
- RSA is the second-largest unique predictor for positive receivers.
- Amino-acid class and normalized position provide almost no unique information.

------------------------------------------------------------------------
4.6 Matched high-versus-low effects
------------------------------------------------------------------------

After exact query-Q8 and distance-bin matching:

Positive-band high receivers versus low receivers:

- Neq: +0.592 to +0.711
- RSA: +0.087 to +0.115
- torsional change: +14.7 to +21.1 degrees
- probability of lying in any band: +18.6 to +32.6 percentage points
- disorder: -0.042 to -0.030
- normalized position: weak and inconsistent

Negative-band high receivers versus low receivers:

- Neq: -0.463 to -0.397
- RSA: -0.136 to -0.103
- torsional change: -38.0 to -34.1 degrees
- probability of lying in any band: +10.1 to +18.0 percentage points
- normalized position: weak and inconsistent

Conclusion:

- Positive bands preferentially influence flexible, exposed and torsionally
  changing queries.
- Negative bands preferentially influence rigid, buried and torsionally stable
  queries.
- High receivers are more likely to lie in another band interval, but dense
  and overlapping band intervals prevent interpreting this alone as a discrete
  band-to-band signaling network.

------------------------------------------------------------------------
4.7 Long-range receivers
------------------------------------------------------------------------

Long range is defined as at least 21 residues from the source-band interval.

Fraction of test high receivers at long range:

Source mechanism              Negative source      Positive source
-------------------------------------------------------------------
Combined                         55.4–60.4%           59.4–64.0%
Evidence-dominated              66.6–69.4%           74.4–81.6%

Mean fraction of directional contribution mass at long range:

- negative combined: 65.9%
- negative evidence-dominated: 70.2%
- positive combined: 71.0%
- positive evidence-dominated: 79.3%

A majority of high receivers and directional contribution mass can therefore
be distant in sequence. Evidence-dominated bands show the greatest long-range
fractions.

These are absolute fractions, not enrichment relative to the number of
available long-range query residues.

------------------------------------------------------------------------
4.8 Outputs
------------------------------------------------------------------------

Output root:

results/publication_comparable_v2/
  analysis_seed_averaged_band_query_receivers/<condition>/

Important outputs:

- per_seed_query_profile_cache/
- band_query_profiles/
- band_query_pairs.csv.gz
- receiver_feature_effects_by_protein.csv.gz
- receiver_feature_summary.csv
- receiver_model_performance.csv
- receiver_feature_ablation_performance.csv
- long_range_receiver_summary.csv
- extraction_audit.json
- parameters.json

------------------------------------------------------------------------
4.9 Conclusion and limitations
------------------------------------------------------------------------

The model does not distribute a band’s contribution uniformly across queries.
After controlling for source band, query Q8 and sequence distance, query
biophysics strongly identifies high receivers.

Neq is the largest unique predictor for both signs. Positive bands
preferentially influence flexible and exposed queries, whereas negative bands
preferentially influence rigid and buried queries. Much of the influence is
long range in sequence.

Limitations:

- No pairwise C-alpha distance, direct contact, network path, structural
  community, domain or strain-covariance feature was used.
- The current analysis cannot distinguish direct 3D contact, contact-network
  transmission, solvent-mediated propagation or a learned flexibility pattern.
- Long sequence distance does not imply long spatial distance.
- The six model conditions use the same proteins and are not independent
  biological datasets.
- The results are predictive associations and do not establish causal
  biochemical transmission.


========================================================================
PHASE 5: SIGNED-BAND SEQUENCE MOTIFS AND PWMs
========================================================================

Scientific questions:

1. Does sequence around a positive or negative band differ from same-protein,
   same-Q8 non-band sequence?
2. Do train-discovered motifs replicate in validation and test proteins?
3. Does sequence improve held-out band-selection prediction beyond Phase 3A
   biophysical features?

Inputs:

- Final averaged bands:
  results/publication_comparable_v2/
    analysis_seed_averaged_signed_bands/signed_bands.csv

- Residue sequence and Q8 annotations:
  results/publication_comparable_v2/
    analysis_seed_averaged_band_biophysics/annotations/
      residue_biophysical_annotations.csv.gz

- Phase 3A Q8-segment candidates:
  results/publication_comparable_v2/
    analysis_seed_averaged_band_phase3a/q8_segment_candidates.csv.gz

------------------------------------------------------------------------
5A: APEX-CENTERED PWM AND LOCAL MOTIFS
------------------------------------------------------------------------

Script:

signed_band_analysis/analyze_signed_band_apex_pwm.py

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

------------------------------------------------------------------------
5B: COMPLETE Q8-SEGMENT MOTIF ANALYSIS
------------------------------------------------------------------------

Script:

signed_band_analysis/analyze_signed_band_sequence_motifs.py

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

Model                              Positive AUROC    Negative AUROC
------------------------------------------------------------------
Q8 + biophysical features              0.812             0.818
Motif features only                    0.557             0.681
Biophysical features + motifs          0.812             0.808

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

------------------------------------------------------------------------
PHASE 5 CONCLUSION
------------------------------------------------------------------------

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

========================================================================
CURRENT OVERALL CONCLUSIONS
========================================================================

1. Signed influence bands are highly reproducible across seeds, and their
   locations are strongly nonrandom relative to protein-preserving
   circular-shift nulls.

2. Positive bands preferentially occupy flexible, exposed, loop-like and
   torsionally active environments. Negative bands preferentially occupy
   rigid, structured, buried and torsionally stable environments.

3. In the 208 test proteins with strain data, positive bands occupy higher and
   more rapidly changing strain environments, whereas negative bands occupy
   lower-strain environments.

4. These differences remain substantial after comparison with same-protein
   residues from the same Q3 class. Band selection therefore reflects more
   than Q3 identity alone.

5. Positive bands are enriched for Q8 loops, Neq peaks and high-strain
   environments, but have low exact recall. Bands should not be interpreted as
   general loop, Neq-peak or high-strain detectors.

6. Selection of a complete Q8 segment is predictable on held-out proteins from
   Neq, exposure, length, geometry, boundary organization and disorder.

7. Experimental structure adds information beyond the predicted biophysical
   features. Positive bands favor weakly connected, mechanically deformable
   C/S/T segments, whereas negative bands favor densely connected and central
   H/E segments resembling stabilizing structural cores.

8. Selected segments are not generally enriched at contact-community or
   ECOD-domain boundaries. Their signal is better described by local packing
   and mechanics than by generic hinge or domain-boundary localization.

9. Positive bands usually combine strong intrinsic signed evidence with broad
   attention consultation. Negative bands are more often
   intrinsic-evidence-dominated, with attention acting as a smaller amplifier.

10. A band’s contribution is not distributed uniformly across query residues.
    After controlling for source band, query Q8 and sequence distance, query
    biophysics strongly predicts high receivers. Neq is the largest unique
    predictor for both signs.

11. Positive bands preferentially influence flexible and exposed queries,
    whereas negative bands preferentially influence rigid and buried queries.
    Much of this influence is long range in sequence, especially for
    evidence-dominated bands.

12. Positive bands have reproducible polar, turn-associated and
    hydrophobic-depleted sequence context. Negative bands have reproducible
    hydrophobic and helix-compatible sequence context.

13. Broad sequence chemistry replicates better than exact short motifs. Motifs
    add essentially no positive held-out discrimination and reduce average
    negative discrimination after the Phase 3A biophysical features are
    included. The results do not support a universal independent sequence code
    for band selection.

14. Overall, the model appears to learn a distributed mechanism in which
    structurally and mechanically distinctive source regions provide signed
    evidence, attention controls how broadly that evidence is consulted, and
    query biophysics helps determine which residues receive it most strongly.

========================================================================
CURRENT LIMITATIONS AND NEXT PHASES
========================================================================

- NetSurfP annotations are predicted structural annotations. Experimental PDB
  geometry and contact networks provide complementary evidence, but structural
  coverage is incomplete for some segments and is somewhat lower for positive
  loop segments.

- A sensitivity analysis restricted to fully resolved experimental segments is
  still recommended.

- Strain is available only for the 208 test proteins. Its associations have not
  yet been replicated using independent train or validation strain datasets,
  and strain was not used as a train-learned incremental predictor.

- Phase 3A and Phase 3B treat complete Q8 segments as candidate objects. This is
  a deliberate estimand, not proof that every contribution band corresponds
  exactly to one Q8 segment.

- Broad and frequently overlapping positive and negative band intervals should
  not be treated as independent physical domains or summed as nonoverlapping
  sequence coverage.

- ECOD boundaries are annotations rather than experimentally established hinge
  axes. Contact-network measurements also depend on structural completeness
  and the selected contact definition.

- Phase 4 controls for sequence distance but does not yet include pairwise 3D
  distance, direct contacts, contact-network paths, structural communities or
  strain covariance between a source band and its receivers. Long sequence
  distance must not be interpreted as long spatial distance.

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

- Phase 6 final cross-phase robustness, sensitivity analysis and integrated
  publication-level synthesis remains to be completed.