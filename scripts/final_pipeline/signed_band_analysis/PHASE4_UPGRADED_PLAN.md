# Phase 4 upgraded: structural and crystallographic-water receiver analysis

## Scientific estimand

For a fixed signed source band \(b\), Phase 4 asks why query residue \(i\)
receives a high sign-aligned contribution

\[
R_{i,b} = \operatorname{sign}_b \sum_{j \in b} C_{ij}
\]

relative to low receivers in the same protein, source band, query-Q8 class,
and sequence-distance bin. The upgrade tests whether experimental 3D
proximity, residue-contact topology, domain/community organization, polar
geometry, or resolved crystallographic-water geometry explains receiver
selection beyond query biophysics.

The analysis is predictive and associative. Attention is not a physical
trajectory, and a water path in one static coordinate model is not evidence of
causal signal transmission.

## Scientific corrections to the initial proposal

1. **Do not call distance-only N/O/S contacts hydrogen bonds.** Most structures
   lack hydrogens and assigned protonation states. Outputs use
   `putative_polar_contact` and `water_path`, with no donor–acceptor angle
   claim.

2. **Use fixed cohorts for sequential AUROC comparisons.** The ordinary
   structure stages use one mapping/resolution-coverage cohort. The water
   progression repeats every earlier stage on one primary-water-eligible
   cohort. Thus a later-stage AUROC change cannot arise merely because
   lower-quality structures disappeared.

3. **Treat unresolved structure as missing, not absence.** A missing query,
   insufficiently resolved source band, unavailable ECOD label, or rejected
   water model produces `NaN`. Observed graph disconnection is encoded as
   `path_exists = 0` with a missing conditional path length.

4. **Separate ordinary and nonlocal contact paths.** The ordinary graph uses
   all 8 Å C-alpha contacts. The nonlocal graph removes edges with
   \(|i-j| < 3\), preventing a shortest path from being almost entirely a
   description of backbone adjacency.

5. **Define water paths without intervening protein nodes.** The water graph is
   residue–water, water–water, and water–residue only. A query-to-band water
   path cannot hop through a third protein residue.

6. **Do not assign waters by their PDB chain identifier.** Primary analysis
   retains waters contacting the mapped target chain and excludes waters also
   contacting another crystallographic chain. This conservative policy avoids
   silently turning asymmetric-unit interface waters into chain-local waters.

7. **Treat ECOD absence as unknown.** “Same domain” is evaluated only when both
   sides have mapped ECOD labels. No label is not encoded as a different
   domain. Domain-boundary distances are explicitly sequence distances in the
   mapped coordinate system.

8. **Make community summaries tie-safe.** The band-apex community and unique
   majority community are separate features. A tied band-community vote is
   missing rather than arbitrarily broken.

9. **Do not interpret the source-mechanism main effect.** Receiver classes are
   balanced within source band, so the mechanism label is constant within the
   primary contrast. The upgraded final stage tests prespecified
   mechanism-by-query-feature interactions; a pooled mechanism main effect is
   not treated as biological evidence.

10. **Defer “independent water paths.”** Counting topologically independent
    paths requires a specific node/edge-disjoint definition and is expensive
    for tens of millions of pairs. The initial pipeline instead reports shared
    waters, shortest water count, and the number of query-contact waters that
    can reach the band water set. Exact disjoint-path analysis can follow if
    the primary water result warrants it.

## Primary feature definitions

### Mapping and coverage

- Mapping identity at least 0.90.
- Model-sequence input coverage at least 0.80.
- Query residue resolved.
- Source-band resolved fraction at least 0.80.
- All coordinates are model/input-sequence indexed and 0-based.

### Ordinary structure

- Minimum query-to-band C-alpha distance.
- Direct 8 Å C-alpha contact, contacted-band-residue count and fraction.
- Shortest ordinary contact-network path.
- Shortest nonlocal contact-network path after removing \(|i-j| < 3\) edges.
- Same nonlocal-contact community as the band apex.
- Same community as the band’s unique majority community.
- Graph distance from the query to a community-boundary node.
- Same ECOD domain as the apex and any shared ECOD domain with the band.
- Query distance to an ECOD boundary and to a boundary of a band-associated
  ECOD domain; both are sequence distances.
- Minimum all-heavy-atom and N/O/S polar-atom distances.
- Local backbone-tangent orientation relative to the band apex.

For a resolved query inside its source band, minimum distance and graph path
are legitimately zero because the query belongs to the target set. Direct
contact still excludes self-contact.

### Putative polar geometry

- Minimum N/O/S query-to-band distance.
- Number and presence of band residues with an N/O/S heavy-atom pair within
  3.5 Å.

These are permissive geometric contacts, not assigned hydrogen bonds.

### Primary crystallographic-water network

- X-ray diffraction structures only.
- Resolution no worse than 2.5 Å.
- Water occupancy at least 0.50.
- Water B factor no more than three robust scaled-MAD units above the mapped
  target protein-atom median.
- Protein-polar-atom–water cutoff of 3.4 Å and water–water cutoff of 3.2 Å.
  The slightly broader protein–water shell tolerates coordinate uncertainty;
  the water–water edge remains closer to a conventional O···O hydrogen-bond
  shell.
- Waters contacting another crystallographic chain are excluded.

Features:

- Query-contact and band-contact water counts.
- Shared-water count, query/band-normalized shared-water fractions, and
  one-water bridge.
- Shortest query-to-band water path, expressed as number of water molecules.
- Path using at most one, two, or three waters.
- Water path remaining when direct C-alpha or direct polar contact is absent.
- Number of query-contact water entry nodes connected to the band water set.
- Mean water-network degree around the query.

The strict and relaxed definitions in `run_phase4_upgraded.sh` are
prespecified sensitivity analyses and are opt-in.

## Held-out model progression

Every model is fitted on train proteins only and evaluated without refitting
on validation and test proteins.

Within each fixed cohort and receiver scope:

1. Sequence distance and query-Q8 baseline.
2. Query biophysics: Neq, RSA, torsion, disorder, amino-acid class, position,
   and membership in any signed band.
3. Experimental 3D geometry.
4. Contact-network features.
5. Contact community and ECOD features.
6. Putative direct polar contacts.
7. Crystallographic-water features.
8. Source mechanism and mechanism-by-query-feature interactions.

The full progression is repeated for:

- all eligible queries;
- sequence-distal queries at least 21 residues from the band;
- in the water cohort, sequence-distal queries with no direct C-alpha contact.

Group ablations compare water features with query biophysics and ordinary
structure on the same water-eligible rows. `full_without_rsa` directly tests
whether RSA retains unique held-out information after water geometry enters.

## Pipeline

1. `python -m signed_band_analysis build-query-structure`
   reads the Phase 3B contact JSON and cached PDB files, constructs the graphs,
   and writes compact per-protein NPZ partitions.

2. `python -m signed_band_analysis audit-phase4-upgraded`
   checks feature identity/shape, missing-data rules, contact/path consistency,
   and water-path invariants.

3. `python -m signed_band_analysis query-receivers --aggregate_only`
   reuses the completed Phase 4 three-seed receiver caches, joins one feature
   partition at a time, repeats matched effects and held-out models, and writes
   a new upgraded result tree.

4. `python -m signed_band_analysis audit-phase4-upgraded` runs again with receiver
   output directories and
   checks required outputs, held-out-only evaluation labels, and constant row
   counts across stages of each fixed cohort.

`signed_band_analysis/run_phase4_upgraded.sh` wires these steps together. It
has not been run.

Run the prespecified primary analysis with:

```bash
bash signed_band_analysis/run_phase4_upgraded.sh
```

After the primary analysis, run only the strict and relaxed sensitivity suite
with:

```bash
RUN_PHASE4_PRIMARY=0 RUN_PHASE4_WATER_SENSITIVITY=1 \
  bash signed_band_analysis/run_phase4_upgraded.sh
```

## Inputs

- `results/publication_comparable_v2/analysis_seed_averaged_signed_bands/signed_bands.csv`
- `results/publication_comparable_v2/analysis_seed_averaged_signed_bands/signed_band_protein_summary.csv`
- `results/publication_comparable_v2/all_split_signed_contributions_manifest.tsv`
- `results/publication_comparable_v2/analysis_seed_averaged_band_biophysics/annotations/residue_biophysical_annotations.csv.gz`
- `results/publication_comparable_v2/analysis_seed_averaged_band_phase3c/mechanism_by_band_seed_averaged.csv.gz`
- Phase 3B `train/validation/test_contacts_ca8.json.gz`
- PDB files referenced by those contact records
- `data_splits/atlas_grouped_v1/ecod_v285_annotations.csv`
- Existing per-condition Phase 4 receiver cache directories

## Outputs

Feature-store root:

`results/publication_comparable_v2/analysis_seed_averaged_band_query_receivers_upgraded/feature_store_primary/`

- `pair_features/<condition>/<split>/<protein>.npz`
- `pair_feature_manifest.csv`
- `structure_water_input_audit.csv`
- `parameters.json`

Per-condition upgraded receiver root:

`.../analysis_seed_averaged_band_query_receivers_upgraded/primary/<condition>/`

- `band_query_pair_manifest.csv`
- `receiver_aggregate_checkpoints/<condition>/<split>/<protein>/pairs.csv.gz`
- `receiver_aggregate_checkpoints/<condition>/<split>/<protein>/complete.json`
- `receiver_feature_effects_by_protein.csv.gz`
- `receiver_feature_summary.csv`
- `receiver_model_performance.csv`
- `receiver_feature_ablation_performance.csv`
- `receiver_structural_water_ablation_performance.csv`
- `long_range_receiver_summary.csv`
- `parameters.json`
- `extraction_audit.json`
- `receiver_complete.json`

The per-protein pair partitions and compact effect/model checkpoints are
written atomically. A rerun with identical inputs and analysis parameters
reuses every completed protein checkpoint, and a completed condition is
skipped using `receiver_complete.json`. The partition manifest is the
authoritative pair-table index; a legacy monolithic `band_query_pairs.csv.gz`
is not required for analysis or audit. A per-output advisory lock prevents
two concurrent receiver processes from corrupting the same checkpoint set.

Top-level audits:

- `audit_primary_features.json`
- `audit_primary_complete.json`

No scientific result should be interpreted unless the final audit passes.
