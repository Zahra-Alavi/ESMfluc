# Publication-v1 attention analyses

These scripts were used with
`results/publication_comparable_v1` and
`results/publication_comparable_v1_attention_sources`. They predate the grouped
train/validation/test split and have not been validated with v2 outputs.

Run scripts from the pipeline root:

```bash
python3 legacy/publication_v1_analysis/analyze_attention_row_modes.py \
  --result_root results/publication_comparable_v1
```

The files remain together because several scripts share helper functions.

`plot_attention_source_grid.py` and `plot_attention_structure_grid.py` remain at
the pipeline root because the v2 contribution plots import their utilities.
