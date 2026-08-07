# GK mutant analysis using fixed WT* bands

This is the current GK case-study pipeline. The older analysis is still in
`results/weinreb2025_mutants_no_AV/mutation_network_analysis/`; it has not been
deleted or overwritten.

The main question is simple: do mutations perturb regions that the trained
model already identifies as important in WT*? Bands are therefore detected
once in the unmutated WT* sequence. Mutant profiles are compared with those
fixed WT* bands. Bands are never detected again in each mutant.

## Inputs

- Forty sequences: WT, WT*, 34 Weinreb mutants, A175G, A175P, A176G and the
  A175G/A176G double mutant.
- Six BiLSTM-attention model conditions, each with seeds 1, 2 and 3.
- Exact contribution matrices and flexible-class probabilities from v2.
- The 1ZNX structure and the checked FASTA-to-PDB position map.
- The published activity measurements for the 34 Weinreb mutants.

A176G was missing from the original contribution files. Its 18 outputs are in
`results/weinreb2025_mutants_no_AV/exact_contributions_a176g_v2/`. They are
combined in memory with the original outputs; the original files are unchanged.

## Fixed bands

The locked Phase 1 detector is applied to WT* separately for each seed and to
the three-seed mean. It uses the raw residue profile, a two-MAD apex threshold,
half-peak-height support, interval IoU of at least 0.5 and support from at least
two seeds. The primary model is `esm3_top28_bilstm_attn`. The other five model
conditions are robustness checks.

The main WT* band table is:

`results/weinreb2025_mutants_no_AV/mutation_network_analysis_v2/reference_bands/primary_wt_star_bands.csv`

Mutation membership is in:

`results/weinreb2025_mutants_no_AV/mutation_network_analysis_v2/reference_bands/mutation_membership_all_conditions.csv`

Paper/PDB numbering is FASTA position plus one. Only residues present in 1ZNX
have structural coordinates; the band table reports the resolved part and
resolved fraction separately.

## What is measured

For every mutant, model condition and seed, the code subtracts the WT*
contribution matrix from the mutant matrix. It then measures:

- the change at the fixed WT* bands;
- the change outside those bands;
- the query residues that receive changed contribution from each band;
- the change in attention routing and in intrinsic signed evidence;
- the change in the predicted flexible-class probability.

The exact change in influence is split into an intrinsic-evidence part and an
attention-routing part. Reconstruction is checked numerically.

The fixed-band hub test asks whether the WT* bands contain more mutant-induced
change than equally shaped masks shifted to other sequence positions. The
uniform-attention control asks whether stable WT* bands remain when the same
learned evidence is consulted uniformly.

The activity analysis uses the 34 Weinreb mutants. It tests six quantities
chosen before looking at activity results, while accounting for experimental
group and binding strain. A second analysis collapses substitutions at the same
position. These are small-sample association tests, not causal tests and not a
new activity predictor.

B1 is A176G. C1 is A175G/A176G. Their published activities come from a different
study and experimental context, so their model outputs are compared
descriptively rather than pooled with the 34-mutant activity regression.
The pipeline also calculates the double-mutant interaction residual directly:
`C1 - A175G - A176G + WT*`. This distinguishes a non-additive model response
from the larger perturbation expected simply because C1 has two substitutions.

## Run

From the repository root:

```bash
bash gk_mutants/run_gk_v2.sh
```

The runner reuses valid A176G extraction files, rebuilds WT* bands and the
uniform control, performs the complete analysis, and finishes with an
independent audit. Results go to:

`results/weinreb2025_mutants_no_AV/mutation_network_analysis_v2/`

The final audit is:

`results/weinreb2025_mutants_no_AV/mutation_network_analysis_v2/audit/final_integrity.json`

The Phase 1 seed-reproducibility calls use one circular shift only because the
shifted null is not used to make a claim for this single-protein case study.
The separate fixed-band hub test evaluates all 207 possible circular shifts
exactly. The 10,000 permutations in the runner belong to the activity tests.

## Interpretation limit

This analysis can show that mutations change model contribution patterns at
fixed WT* regions. It cannot by itself prove that a band is a physical
allosteric pathway. Structural proximity and experimental activity are
supporting comparisons, not causal validation.
