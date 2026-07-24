# GK mutant contribution-network analysis

**State summarized:** 2026-07-24

## 1. Purpose

This project asks whether mutations in *Mycobacterium tuberculosis* guanylate
kinase (GK) reproducibly reorganize the residue-to-residue contribution network
learned by ESMfluc, including changes far from the mutation, and whether those
changes relate to experimental catalytic activity or mechanical measurements.

The analysis does **not** treat an attention or contribution pattern as a
physical allosteric pathway by itself. It first establishes seed
reproducibility, then separates intrinsic residue evidence from attention
routing, maps changed keys to the query residues that receive their
contributions, and finally tests associations with experimental phenotypes.

## 2. Sources of the mutants

### Weinreb et al., Nature Physics (2025)

The main experimental dataset comes from:

- E. Weinreb et al., “Enzymes as viscoelastic catalytic machines,” *Nature
  Physics* 21, 787–798 (2025).
- DOI: <https://doi.org/10.1038/s41567-025-02825-9>

The paper studied 34 single-residue variants of a cysteine-substituted
experimental construct called **WT\***. The variants were selected as:

- **Binding (9):** S30G, E88N, A31V, S53N, S30Q, R60K, G33S, T101A, V32L.
- **High-Strain (11):** G178S, D179S, A175T, L174F, Q177D, E173N, A176C,
  L174G, P29A, E173H, P29V.
- **Control (14):** A58V, I92V, S193A, P59A, S51G, A127T, G62S, V25A,
  R42H, P46A, V120A, G94S, P61A, I118F.

The local dataset also contains native `WT`. It is retained only as a
background-construct control. All mutation effects for the paper variants are
calculated relative to `WT_star`, not native `WT`.

The paper supplies normalized activity, activity uncertainty and replicates,
binding and mutation strain, experimental group, stability information for a
subset, and nano-rheology measurements for a smaller subset.

### Alavi and Zocchi, Physical Review E (2018)

The second relevant paper is:

- Z. Alavi and G. Zocchi, “Dissipation at the angstrom scale: Probing the
  surface and interior of an enzyme,” *Physical Review E* 97, 052402 (2018).
- DOI: <https://doi.org/10.1103/PhysRevE.97.052402>
- Accepted manuscript:
  <https://link.aps.org/accepted/10.1103/PhysRevE.97.052402>

It studied mutations in the high-strain 175–176 hinge:

- **B1:** A176G in the paper’s/PDB residue numbering, approximately
  0.1 times WT activity.
- **C1:** A175G/A176G, approximately 10 times WT activity.

Thus the additional A175G substitution changes activity by approximately
100-fold when comparing B1 with C1. The paper attributes the speed difference
mainly to `kcat`, rather than a large change in substrate or product affinity.
It reports subtle mechanical differences: C1 has approximately 1.4 times the
fitted internal dissipation of WT. It also proposes that residues 175–176,
which lie near an internal water channel, could alter closed-state hydration.
C1 shows a possible second, low-affinity GMP-binding event, suggesting a
possible speed–specificity trade-off.

### Additional local sequences and their current status

The current FASTA contains 40 records:

- native `WT`;
- `WT_star`;
- the 34 Weinreb paper mutants;
- `A175G`, `A175P`, `A175G_A176G`, and `A176G`.

The completed contribution-network analysis predates the addition of `A176G`.
It covers 39 records: native WT, WT\*, 34 Weinreb mutants, `A175G`, `A175P`,
and `A175G_A176G`. `A176G` is present in the FASTA now but is not present in
the ESM3-top28 seed outputs and is therefore not in the current derived
analysis.

`A175P` is a locally added chemical/structural comparison, not an experimental
mutant reported in either of the two papers above. The single `A175G` sequence
is useful for separating the two substitutions in C1, but the 2018 paper did
not report it as B1.

## 3. Numbering and B1/C1 integrity

The protein arrays have length 207. Where a PDB coordinate exists:

```text
paper/PDB residue = FASTA position + 1
matrix index       = FASTA position - 1
```

The local mutation labels use paper/PDB-style numbering even though the changed
array element is one position lower. Direct sequence comparison against
`WT_star` gives:

| Local record | Changed FASTA positions | Corresponding PDB residues |
|---|---:|---:|
| `A175G` | 174 | 175 |
| `A176G` | 175 | 176 |
| `A175G_A176G` | 174 and 175 | 175 and 176 |

Therefore the coordinate-verified mapping is:

```text
B1 = A176G
C1 = A175G_A176G
```

Attaching B1’s 0.1-times-WT activity to the analyzed `A175G` record would be a
phenotype-to-sequence mismatch. `A176G` must first be processed through the
same model conditions and seeds before a valid B1–WT\*–C1 network comparison
can be made.

Only 182 of 207 positions have usable C-alpha coordinates. The remaining 25
positions are retained in sequence analyses but explicitly reported as
structurally unmapped.

## 4. Inputs and model conditions

Important inputs are:

- FASTA: `data/weinreb2025_mutants_no_AV.fasta`
- Contribution root:
  `scripts/final_pipeline/results/weinreb2025_mutants_no_AV/exact_contributions_v2`
- Closed structure: PDB 1ZNX
- Open structure: PDB 1ZNW
- Paper activity and nano-rheology data: Supplementary Data 1 from Weinreb
  et al.

The primary, pre-specified model condition is:

```text
esm3_top28_bilstm_attn, seeds 1, 2 and 3
```

Five other conditions are used only for robustness after the primary bands
and hotspot are frozen:

- ESM3 top-4, ESM3 frozen;
- ESM2 top-28, ESM2 top-4, ESM2 frozen.

## 5. Contribution quantities

For mutant \(m\), seed \(s\), query residue \(i\), and key residue \(j\), the
signed contribution matrix is \(C^{(s)}_{m,ij}\). A positive contribution
supports flexibility and a negative contribution supports rigidity.

The importance of key \(j\), averaged over all queries, is:

\[
I^{(s)}_{m,j}=\frac{1}{L}\sum_i C^{(s)}_{m,ij}.
\]

The seed-matched mutation effect is:

\[
\Delta I^{(s)}_{m,j}
=I^{(s)}_{m,j}-I^{(s)}_{WT_\star,j}.
\]

The pipeline retains the WT\* baseline \(I\), because the meaning of
\(\Delta I\) depends on baseline sign. For example, positive \(\Delta I\) at a
negative WT\* baseline can mean weaker rigidity support, not necessarily a
flexibility-supporting residue.

The matrix orientation is:

```text
row i    = query/receiver residue
column j = consulted key residue
```

Consequently, \(I_j\) identifies changed keys but cannot alone reveal which
query residues receive those changes.

## 6. What was implemented

The analysis was implemented as nine scripts in `gk_mutants`:

1. **`weinreb_analysis_common.py`**
   centralizes model constants, paths, FASTA and structure parsing, coordinate
   maps, domain annotations, contribution loading, and shared numerical
   utilities.

2. **`build_weinreb_delta_profiles.py`**
   validates record presence, sequences, \(207\times207\) matrices, finite
   values, attention normalization and exact contribution reconstruction. It
   builds seed-matched WT\* and \(\Delta I\) profiles with mean, SD, SE, sign
   agreement and effect-to-seed-variation.

3. **`detect_weinreb_change_bands.py`**
   detects contiguous positive and negative bands directly in mean
   \(\Delta I\). It applies seed reproducibility filters, a 27-setting
   sensitivity grid, sequence and C-alpha distance calculations, normalized
   distal burden, and a genome-wide common-hotspot test.

4. **`decompose_weinreb_contributions.py`**
   uses \(I_j=s_j\bar A_j\) to split each change exactly into:

   \[
   \Delta I_j =
   \frac{\bar A_m+\bar A_{WT_\star}}{2}(s_m-s_{WT_\star})
   +
   \frac{s_m+s_{WT_\star}}{2}(\bar A_m-\bar A_{WT_\star}).
   \]

   The first term is changed intrinsic signed evidence. The second is changed
   attention consultation/routing.

5. **`map_weinreb_receivers.py`**
   returns to the full matrix. For each changed key band \(B\), it calculates:

   \[
   \Delta C_{i,B}=\sum_{j\in B}
   \left(C_{m,ij}-C_{WT_\star,ij}\right).
   \]

   It identifies query residues receiving the change, tests structural and
   functional enrichment with matched permutations, and compares each full
   query-routing pattern with both seed-matched WT\* and the typical pattern
   across other mutants.

6. **`analyze_weinreb_associations.py`**
   joins paper activity and rheology data and constructs interpretable ESM
   summary variables. For activity, each ESM variable is tested separately
   with weighted regression, paper covariates, residual permutation tests,
   FDR correction, and leave-one-mutant-out prediction. A second model asks
   whether topology adds information beyond ordinary predicted
   \(N_\mathrm{eq}\) change.

   For mechanics, it measures each mutation-averaged phase/amplitude curve’s
   distance from WT\* and uses an exact High-Strain versus Binding/Control
   label-permutation test. This is an indirect group comparison, not a direct
   mutant-by-mutant regression of ESM changes against rheology.

7. **`replicate_weinreb_conditions.py`**
   tests the 92 primary bands and hotspot in the five alternate
   model/freezing conditions without retuning.

8. **`audit_weinreb_analysis.py`**
   checks output completeness, row counts, numerical reconstruction and
   cross-file consistency.

9. **`run_weinreb_mutation_network_analysis.sh`**
   executes the complete pipeline with location-independent paths.

All derived results remain under:

```text
results/weinreb2025_mutants_no_AV/mutation_network_analysis
```

### Running the analysis

Run the complete pipeline from the `final_pipeline` directory:

```bash
bash gk_mutants/run_weinreb_mutation_network_analysis.sh
```

The runner and Python defaults resolve paths from the `gk_mutants` directory,
so the runner can also be launched through its absolute path from another
working directory without changing input or output locations.

Paper inputs are retained in the output tree's `inputs` directory, with hashes
recorded in `audit/final_integrity.json`. The coarse domain boundaries used for
stratification are defined in `gk_mutants/weinreb_analysis_common.py`; they are
annotations for matched comparisons, not claims of atomically exact domain
boundaries.

## 7. Main results

### Integrity and reproducibility

- The completed snapshot contains 39 records in each of three primary seeds.
- All 117 primary contribution matrices are \(207\times207\), finite,
  sequence-matched and reconstruct correctly.
- Maximum contribution/logit reconstruction error:
  \(7.66\times10^{-7}\).
- Maximum decomposition reconstruction error:
  \(1.10\times10^{-9}\).
- The conservative primary detector retained 92 signed change bands.
- 89 bands have 3/3 integrated seed-sign agreement; three have 2/3.
- Fifty bands are distal by both greater than 15 sequence positions and
  greater than 12 Å, covering 31 non-reference sequences.
- Full-profile seed correlations are modest, with median pairwise correlation
  approximately 0.163. Stable bands are therefore more trustworthy than the
  noisy fine structure of entire profiles.

### Common response hotspots

The preliminary matrix positions 155–157 map to FASTA positions 155–157 and
paper/PDB residues 156–158 in the LID.

- This is the top three-residue window across the protein.
- Genome-wide empirical \(p=0.0097\).
- LID-matched empirical \(p=0.0488\).

It is a recurrent control/response node, not by itself proof of a unique
mutation-specific allosteric pathway. Other recurrent regions occur near paper
residues 94–95 and 45–46.

### Intrinsic evidence versus attention routing

The exact decomposition assigns:

- 69 of 92 bands as attention-routing dominated;
- 23 of 92 as intrinsic-evidence dominated.

Thus most stable mutation-induced bands arise more from changed consultation
of residue keys than from changed intrinsic flexible/rigid evidence at those
keys.

### Receiver and query-pattern analysis

Matched structural permutations find recurring enrichment of changed
contributions at:

- experimental high-strain positions;
- hinges;
- the C75–C171 structural corridor;
- residues with large open-to-closed displacement.

These overlapping enrichments support a candidate mechanical communication
network, but not one uniquely established causal route.

Query behavior is mutant-specific and can be far from the mutation. For
example, S30Q strongly changes paper query 155, approximately 125 sequence
positions and 18 Å from S30. Its largest mutant-specific routing change points
to paper key 156. The same-site S30G substitution produces much less global
and distal query reorganization, showing that substitution chemistry matters.

In the completed 39-record snapshot:

- `A175G` has relatively low overall and distal query rerouting.
- `A175G_A176G` has substantially stronger, mutant-specific long-range
  rerouting, including LID queries 128–130 and 156–157 and a remote response
  near 195.
- At query 129, the double mutant’s logit-margin change is approximately
  \(-0.179\), with its largest mutant-specific routing residual directed toward
  key 157.

These are model-level observations. Because the true B1 sequence `A176G` has
not yet been processed, they cannot yet explain the measured B1-to-C1
100-fold activity difference.

### Activity associations for the 34 Weinreb mutants

Activity was modeled as a continuous WT\*-normalized outcome. Models were
weighted by reported activity uncertainty and adjusted for binding strain,
Binding-group membership and High-Strain-group membership.

The strongest ESM associations are:

| ESM quantity | Spearman rho | Adjusted FDR q | Interpretation |
|---|---:|---:|---|
| Contribution change received by binding residues | -0.661 | 0.0080 | More topology change reaching binding queries is associated with lower activity |
| Distal absolute \(\Delta I\) burden | -0.638 | 0.0213 | More distal reorganization is associated with lower activity |
| Gain of rigidity | -0.541 | 0.0080 | More movement toward rigidity is associated with lower activity |

After additionally controlling for ordinary predicted
\(N_\mathrm{eq}\) change:

- gain of rigidity remains significant:
  partial \(R^2=0.101\), FDR \(q=0.0126\);
- binding-receiver contribution change remains significant:
  partial \(R^2=0.035\), \(q=0.0315\);
- distal burden is borderline after multiplicity correction:
  partial \(R^2=0.045\), \(q=0.0546\).

Leave-one-mutant-out RMSE improvements are small. The results show association
and modest additional information, not strong prospective activity prediction
or causation.

The pipeline does not currently include pure attention total variation as an
activity predictor. A separate read-only exploratory calculation found that
mean, distal and LID query-routing magnitude were also negatively associated
with activity across the 34 Weinreb mutants and retained signal after paper
covariates and \(N_\mathrm{eq}\). Because this follow-up was post hoc and was
not written into the pipeline outputs, it should be treated as a hypothesis for
a pre-specified extension rather than a finalized result.

### Mechanical subset

The pre-specified mechanical subset contains:

- High-Strain: E173N, P29V, A175T, D179S;
- Control: G62S, V120A and WT\*;
- Binding: S30Q, G33S, R60K.

For the nine non-WT mutants, exact tests found no significant High-Strain
versus Binding/Control separation:

- experimental rheology-curve distance: \(p=0.748\);
- ESM distal burden: \(p=0.654\).

The current script does not directly regress mutant-level ESM topology against
mutant-level rheology and does not use a repeated-measurement model over
frequency and experimental day. The null group test therefore does not prove
that ESM changes and mechanics are unrelated.

### Cross-condition robustness

The 92 ESM3-top28 primary bands were frozen and tested without retuning:

| Alternate condition | Same-sign, at least 2/3-seed replication |
|---|---:|
| ESM3 top-4 | 67.4% |
| ESM3 frozen | 62.0% |
| ESM2 frozen | 54.3% |
| ESM2 top-4 | 53.3% |
| ESM2 top-28 | 51.1% |

Flattened profile correlations with the primary condition are only moderate.
The reproducible primary signal is meaningful, but the exact residue-level
pattern is not fully model-condition independent.

## 8. What the results support

The completed analysis supports the following restrained conclusions:

1. Mutations reproducibly reorganize signed contribution bands at both local
   and structurally distant residues.
2. A recurrent LID response around paper residues 156–158 is unusually strong
   across mutants.
3. Most retained band changes are dominated by changed attention
   consultation/routing.
4. Changed keys influence distant query residues in a
   substitution-specific manner.
5. Several topology summaries associate with experimental activity beyond
   ordinary predicted flexibility changes.
6. The small mechanical-subset group test is null, and cross-condition
   replication is moderate.

Therefore the results identify a plausible LID/GMP-binding/hinge communication
network and testable mutant-specific hypotheses. They do not yet establish a
single causal allosteric pathway or prove that ESM attention corresponds
directly to physical force transmission.

## 9. Required next steps for B1 and C1

Before using the 2018 activity measurements in the phenotype analysis:

1. Process `A176G` (B1) through all required model conditions and seeds.
2. Rebuild and audit the analysis with 40 records.
3. Compare seed-matched WT\*, B1/A176G and C1/A175G_A176G directly.
4. Separate magnitude from direction of LID/query rerouting.
5. Test whether the B1-to-C1 activity ordering follows a specific routing
   pattern rather than total network-change burden.
6. Confirm that the experimental reference constructs and numbering are
   sequence-compatible before joining 2018 phenotypes to the 2025 WT\*
   analysis.

For physical validation, replicated long-timescale or enhanced-sampling MD
should compare open/closed LID populations, closure free energy and kinetics,
hinge/water-channel contacts, ligand geometry, and dynamical-network paths
between residues 175–176 and predicted remote responses such as 129, 156–158
and 195. Repeated trajectories are preferable to relying on one long
trajectory, and the physical analysis should test pre-specified ESM hypotheses
rather than attempting to equate attention weights directly with forces.

## 10. Key result files

- `audit/final_integrity.json`: completed-snapshot integrity.
- `profiles/delta_I_summary.csv`: per-mutant, per-residue WT\* baseline and
  seed-averaged \(\Delta I\).
- `bands/change_bands_primary.csv`: retained signed mutation-induced bands.
- `bands/distal_burden.csv`: distance-thresholded mutation burden.
- `hotspots/common_hotspots.csv`: recurrent response positions.
- `decomposition/band_components.csv`: intrinsic versus routing components.
- `receivers/band_receivers.csv`: queries receiving each changed key band.
- `receivers/query_routing_summary.csv`: full seed-averaged query-pattern
  changes.
- `receivers/query_mutant_specificity.csv`: query changes beyond the typical
  response across mutants.
- `phenotypes/esm_predictors.csv`: interpretable topology summaries per
  mutant.
- `phenotypes/activity_associations.csv`: adjusted activity tests.
- `mechanics/mechanical_subset_summary.csv`: rheology distance and ESM burden.
- `mechanics/mechanical_subset_exact_tests.csv`: exact small-subset tests.
- `robustness/replication_summary.json`: alternate-condition replication.
