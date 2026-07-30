# Signed-band analysis package

This package contains analyses of positive and negative signed column-influence
profiles (`I_j`).

## Current Phase 1 method

### 1. Detect candidate apices

The detector is applied separately to every seed profile and to the
three-seed mean profile. A candidate \(p\) must be a raw positive local maximum
or raw negative local minimum satisfying

\[
R_p=\frac{|I_p|}{\sigma_{\mathrm{MAD}}}\ge 2,
\qquad
\sigma_{\mathrm{MAD}}
=1.4826\,\operatorname{median}(|I-\operatorname{median}(I)|).
\]

The scale is calculated once from the eligible raw profile. There is no
smoothing requirement and prominence does not control detection. Prominence
is stored only as a descriptive shape measurement.

By default, residues `1 ... L-2` in 0-based coordinates are eligible. Only the
two protein endpoints are excluded, and they are excluded exactly once.

### 2. Define and merge bands

Starting at each candidate apex, the band extends through contiguous residues
of the same sign satisfying

\[
|I_j|\ge 0.5|I_p|.
\]

Overlapping same-sign half-intensity intervals are merged. A merged band keeps
the strongest candidate as its primary apex and records all candidate apices
that were merged into it. The interval and primary apex are therefore distinct
pieces of information.

Each row stores the signed and absolute apex influence, \(R_p\), integrated
absolute band influence, fraction of eligible total absolute influence,
prominence, and within-protein ranks.

### 3. Determine seed stability

The authoritative objects are bands detected from the three-seed mean profile.
Each mean-profile band is matched one-to-one to same-sign per-seed bands using
interval intersection-over-union:

\[
\operatorname{IoU}(a,b)
=\frac{|a\cap b|}{|a\cup b|}.
\]

A seed supports the mean band when `IoU >= 0.5`. A band is:

- primary stable when supported by at least 2 of 3 seeds;
- strict stable when supported by all 3 seeds.

This interval rule allows a reproducible band to remain stable when merging
causes different seeds to retain different primary apices. Apex positions and
displacements remain descriptive metadata and do not determine stability.

The block-shift null moves each complete same-sign band pattern around the
eligible sequence circle. Every apex and support residue receives the same
modular offset. Boundary-crossing support is represented internally by two
linear segments, so band count, width, sign, magnitude, and circular spacing
remain unchanged. With `--n_block_shifts 1`, the command runs in smoke-test
mode and reports sample standard deviations and z-scores as NA.

The old multiscale-prominence detector, half-prominence intervals, and
apex-position matching remain available only for legacy comparison and
sensitivity analysis.

Q8 segments are external annotations. They do not determine apex detection,
band boundaries, merging, or seed stability.

Run commands from the repository root:

```bash
python -m signed_band_analysis --help
python -m signed_band_analysis extract-bands --help
python -m signed_band_analysis seed-reproducibility --help
python -m signed_band_analysis build-uniform-control --help
python -m signed_band_analysis model-mechanism --help
```

Phase 1 uses only two commands: `extract-bands` detects per-seed and mean-profile
bands; `seed-reproducibility` matches them and writes `stable_signed_bands.csv`
directly.

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
- post hoc Q8 stratification and external-structure analysis;
- evidence/consultation mechanism analysis;
- uniform-routing/evidence-only control profile generation and seed averaging;
- query-receiver and structural-water analysis;
- apex PWM and complete-segment motif analysis.

The following remain outside this package because they are shared upstream
inputs or general attention tools:

- `Attention/extract_all_split_signed_contributions.py`;
- `Attention/audit_all_split_signed_contributions.py`;
- `Attention/build_contact_maps_from_pdb.py`;
- exact-contribution matrix visualizers in the repository root.

The uniform-control command implements only detector-independent preparation:
it streams `s_j`, `B_j`, and observed `I_j`, validates the decomposition,
writes `I_j_uniform = s_j/L` and `G_j = LB_j`, constructs aligned three-seed
means, and reports per-seed and mean-profile similarity with equal protein
weighting. It does not itself run the band comparison. Its two generated
manifests are accepted directly by `extract-bands`, which validates the compact
control schema before applying the locked detector:

```bash
python3 -m signed_band_analysis extract-bands \
  --manifest_tsv analysis_uniform_attention_control/profiles/profile_manifest.tsv \
  --output_dir analysis_uniform_attention_control/apices/per_seed_uniform \
  --influence_field uniform_signed_influence \
  --apex_method raw_mad_amplitude --amplitude_mad 2 \
  --support_method half_intensity_merge

python3 -m signed_band_analysis extract-bands \
  --manifest_tsv analysis_uniform_attention_control/seed_averaged_profiles/seed_averaged_profile_manifest.tsv \
  --output_dir analysis_uniform_attention_control/apices/mean_uniform \
  --influence_field seed_averaged_uniform_signed_influence \
  --apex_method raw_mad_amplitude --amplitude_mad 2 \
  --support_method half_intensity_merge
```

Use `observed_signed_influence` and
`seed_averaged_observed_signed_influence` for the paired observed calls.

The primary output is `signed_bands.csv`. A merged band retains one primary
apex. To keep existing consumers working, each row contains both the new
`support_*` names and compatibility aliases:

| Canonical field | Also stored as |
|---|---|
| `band_id` | `apex_id` (transitional alias) |
| `support_start_index_0based` | `start_index_0based` |
| `support_end_index_0based_inclusive` | `end_index_0based_inclusive` |
| `support_width` | `band_width` |
| `support_sequence` | `band_sequence` |
| `support_absolute_influence_sum` | `band_absolute_influence_sum` |

Required CLI paths are still
resolved from the caller's working directory. The packaged biophysical runner
changes to the repository root before executing, so it is safe to launch from
any directory:

```bash
bash signed_band_analysis/run_signed_band_biophysical_pipeline.sh
```
