# Phase 4 upgraded static audit

Status: implementation reviewed and revised; helper tests passed; the full
data pipeline was intentionally not executed.

## Audit findings addressed

1. **Hydrogen-bond overclaim**

   Initial geometric N/O/S criteria could be misread as assigned hydrogen
   bonds. Code, schema, and documentation now use `putative_polar_contact` and
   crystallographic `water_path`.

2. **Missing structure encoded as negative evidence**

   Unresolved queries and bands below the resolved-fraction threshold now
   receive `NaN` structural/water features. Observed graph disconnection uses a
   separate `path_exists = 0` indicator and a missing conditional path length.

3. **Backbone-dominated graph paths**

   Both ordinary 8 Å C-alpha paths and nonlocal paths excluding
   \(|i-j| < 3\) edges are emitted.

4. **Changing sample across sequential stages**

   Models now report `all_receivers`, `structure_eligible_fixed`, and
   `water_eligible_fixed` cohorts. Row membership is fixed across every stage
   compared within a cohort.

5. **Crystallographic-chain ambiguity**

   Water chain identifiers are not used for assignment. Primary analysis
   excludes a target-contacting water when it also geometrically contacts a
   different crystallographic chain.

6. **Self-water bridge inside a source band**

   For a query inside its source band, its own residue is removed from the band
   water target before bridge/path calculation. A query water is therefore not
   counted as a bridge back to the identical residue.

7. **Arbitrary community tie breaking**

   Apex and majority community features are separate. A tied majority is
   missing and is accompanied by `band_community_majority_defined`.

8. **ECOD absence treated as a different domain**

   Same-domain indicators are missing unless the relevant residues have mapped
   ECOD labels. Boundary features are explicitly named as sequence distances.

9. **Source-mechanism main-effect interpretation**

   Because high/low receivers are balanced within source band, the mechanism
   main effect is not treated as mechanistic evidence. The final stage adds
   prespecified mechanism-by-query-biophysics interactions.

10. **Water-count exposure bias**

    In addition to raw shared-water counts, the implementation emits fractions
    normalized by query-contact and band-contact water counts.

11. **RSA mediation test**

    The fixed water-cohort ablation table includes
    `full_without_rsa`, ordinary-structure-only, structure-plus-water, and
    leave-one-feature-group-out specifications.

12. **Sequence-distal specificity**

    Matched effects and held-out models are repeated at sequence separation
    \(\ge 21\), and water-cohort models add a sequence-distal/no-direct-C-alpha
    scope.

13. **Output integrity**

    The independent audit checks partition schema, band order, array shape,
    unresolved-value semantics, inside-band zero-distance/path invariants,
    contact-count consistency, water-path logic, fixed row counts across model
    stages, and held-out-only evaluation labels.

14. **Original Phase 4 cache preservation**

    Aggregate-only upgrades read completed receiver profiles from the original
    Phase 4 result tree. If a derived averaged profile must be rebuilt, it is
    written under the new upgraded output rather than back into the original
    result directory.

15. **Moved-package import compatibility**

    Package-command checks exposed the repository environment's
    NetworkX-2.3/NumPy-1.24 alias incompatibility. The structural-water builder
    now applies the same compatibility shim as the established Phase 3B module
    before importing NetworkX. The package dispatcher, all three upgraded
    command entry points, the runner's shell syntax, and all repository tests
    were then checked successfully without running the analysis.

## Deliberately deferred

- Donor–acceptor angular hydrogen-bond assignment requires added hydrogens and
  protonation-state modeling.
- Exact node- or edge-disjoint water paths require a prespecified definition
  and substantially more computation.
- Biological-assembly interfaces are not reconstructed.
- Static crystallographic waters cannot establish solution occupancy,
  residence time, exchange, or causal propagation.

## Files reviewed

- `build_band_query_structural_water_features.py`
- `analyze_signed_band_query_receivers.py`
- `audit_phase4_upgraded.py`
- `run_phase4_upgraded.sh`
- `PHASE4_UPGRADED_PLAN.md`
- `tests/test_phase4_structural_water_features.py`

The executable audit remains the authoritative check after a real run. No
scientific output should be interpreted unless that audit passes.
