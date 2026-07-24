# Signed-band analysis package

This package contains the analyses that begin with signed column influence
profiles (`I_j`) and operate on detected positive or negative bands.

Run commands from the repository root:

```bash
python -m signed_band_analysis --help
python -m signed_band_analysis extract-bands --help
python -m signed_band_analysis model-mechanism --help
```

The package dispatcher keeps internal imports stable while giving each analysis
a short command name. Individual modules can also be imported in Python, for
example:

```python
from signed_band_analysis.extract_signed_contribution_bands import iter_profiles
```

## Scope

Included here:

- seed averaging and band detection;
- seed reproducibility;
- Neq/NetSurfP annotation;
- biophysical enrichment and its audit/figures;
- Q8 object selection and external-structure analysis;
- evidence/consultation mechanism analysis;
- query-receiver and structural-water analysis;
- apex PWM and complete-segment motif analysis.

The following remain outside this package because they are shared upstream
inputs or general attention tools:

- `Attention/extract_all_split_signed_contributions.py`;
- `Attention/audit_all_split_signed_contributions.py`;
- `Attention/build_contact_maps_from_pdb.py`;
- exact-contribution matrix visualizers in the repository root.

All result paths and schemas are unchanged. Required CLI paths are still
resolved from the caller's working directory. The packaged biophysical runner
changes to the repository root before executing, so it is safe to launch from
any directory:

```bash
bash signed_band_analysis/run_signed_band_biophysical_pipeline.sh
```
