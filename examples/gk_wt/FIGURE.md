# Guanylate-kinase example: Figure 3

The example uses the supplied 207-residue guanylate-kinase sequence and ESM3 top-28, seeds 1, 2 and 3. The structural panel uses PDB 1ZNX, chain A. The FASTA record is labeled `WT`.

The shipped influence CSV, band CSV and audit are under `results/gk_wt_figure/`. To regenerate them, restore the checkpoints from Zenodo and keep the training run manifest:

```bash
python -m esmfluc.inference.extract_external_fasta_contributions --fasta_file examples/gk_wt/native_wt.fasta --source_manifest results/manifest.tsv --output_root runs/guanylate_kinase/exact_contributions --conditions esm3_top28_bilstm_attn --seeds 1 2 3
python -m examples.gk_wt.prepare_gk_figure_profiles --manifest runs/guanylate_kinase/exact_contributions/manifest.tsv --fasta-file examples/gk_wt/native_wt.fasta --protein WT --split gk_wt --output-dir runs/guanylate_kinase/profiles
```

The preparation step checks the sequence and exact factorization, then exports per-seed and mean influence profiles. Use the detector and seed-matching commands from the analysis guide with these inputs:

- Per-seed manifest: `runs/guanylate_kinase/profiles/profiles_manifest.tsv`.
- Mean manifest: `runs/guanylate_kinase/profiles/averaged_profiles_manifest.tsv`.
- Raw amplitude threshold: 2 MAD units; no smoothing or terminal exclusion.
- Same-sign half-height intervals, merging overlaps.
- One-to-one interval-IoU matching at 0.5; support from at least two seeds.

After detection and matching, export the heatmap and profile:

```bash
python -m esmfluc.visualization.plot_seed_averaged_contributions --manifest-tsv runs/guanylate_kinase/exact_contributions/manifest.tsv --fasta-file examples/gk_wt/native_wt.fasta --profiles-manifest-tsv runs/guanylate_kinase/profiles/averaged_profiles_manifest.tsv --bands-csv runs/guanylate_kinase/bands/reproducibility/stable_signed_bands.csv --condition esm3_top28_bilstm_attn --protein WT --split gk_wt --output-prefix runs/guanylate_kinase/figure
```

Write the preceding detector outputs under `runs/guanylate_kinase/bands/per_seed`, `mean` and `reproducibility` to use that command unchanged. The exporter writes PDF, editable SVG, influence and band tables, and a source-hash audit, using the saved bands.

The heatmap is the elementwise mean of the three contribution matrices. The influence profile is its mean over query rows. Shading shows saved inclusive band intervals; dots show primary peaks. Colors saturate symmetrically at the 99th percentile of absolute contributions.

The structure panel is rendered separately in ChimeraX. Align the input sequence to 1ZNX chain A before coloring, since input positions and PDB numbers differ. Color positive bands red and negative bands blue, with darker C-alpha spheres at their primary peaks. Assemble the exported plot and structure view in a vector editor.

To explore the sequence and structure interactively, use the notebook viewer.
