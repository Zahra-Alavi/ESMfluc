# ESMfluc

ESMfluc predicts whether a residue samples one or several local Protein Block states. Its attention–classifier stage can be decomposed into signed residue contributions, revealing where evidence for each prediction comes from. These contributions describe the model's calculation, not causal interactions between residues.

## Install

Use Python 3.11 in a new environment, then install from the repository root:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The pinned environment includes both ESM backbones and the structure-analysis dependencies. Model inference needs substantial memory; a GPU is recommended. MMseqs2, ChimeraX and MDStrainMapper are separate tools, not bundled software. See [the analysis guide](docs/analysis_guide.md) for inputs and provenance.

## Predict and explore

The trained models are archived on Zenodo (record 23580685, DOI [10.5281/zenodo.23580685](https://doi.org/10.5281/zenodo.23580685)). To run ESMfluc in the browser, open the notebook in [Google Colab](https://colab.research.google.com/github/Zahra-Alavi/ESMfluc/blob/main/notebooks/ESMfluc.ipynb). To download and restore models locally:

```bash
python release/download_zenodo.py --record 23580685 --condition esm3_top28_bilstm_attn --seeds 1 2 3 --destination models
python release/checkpoint_bundle.py --bundle models --condition esm3_top28_bilstm_attn --seed 1 --output models/seed_1.pth
```

The deposit will contain shared backbones, seed-specific tensor overrides, checksums and archived inference code. Checkpoints are restored losslessly; weights are not averaged. The notebook also needs the deposit's `metadata.zip`, whose checksum is pinned in its runner.

The [Colab notebook](notebooks/ESMfluc.ipynb) accepts a sequence alone or a sequence and PDB structure. Clicking a source residue selects its contribution column. Receiving residues above an adjustable absolute-contribution threshold are highlighted in the sequence and, when supplied, the structure. Three-seed mode detects seed-stable bands; single-seed preview does not establish stability.

With a complete, matching local deposit bundle, the same viewer can be run locally:

```bash
python notebooks/esmfluc_runner.py validate --fasta examples/gk_wt/native_wt.fasta
python notebooks/esmfluc_runner.py predict --bundle models --fasta examples/gk_wt/native_wt.fasta --condition esm3_top28_bilstm_attn --seeds 1 2 3 --device cuda --output runs/guanylate_kinase
python notebooks/esmfluc_runner.py map --output runs/guanylate_kinase --pdb external_inputs/1znx.pdb --chain A
```

The structure is optional and used for display, not prediction. Sequence–structure alignment is checked before mapping. Unresolved coordinates remain missing.

## Train

Reconstruct the fixed ATLAS inputs first, following the analysis guide. Train a condition with:

```bash
esmfluc-train --condition esm2_frozen_bilstm_attn --seed 1 --result_foldername runs/esm2_frozen_bilstm_attn/seed_1
```

Repeat for seeds 1, 2 and 3. The seven conditions are `esm2_frozen_linear`, and `{esm2,esm3}_{frozen,top4,top28}_bilstm_attn`. The fixed split is independent of training seed. The frozen linear baseline has no task-specific attention contribution matrix.

## Reproduce the paper

[Analysis guide](docs/analysis_guide.md) · [Data dictionary](docs/data_dictionary.md) · [Saved result index](results/README.md) · [Guanylate-kinase example](examples/gk_wt/FIGURE.md).

Compact derived results are included; trajectories, raw third-party exports, weights and regenerable run arrays are not. A published deposit and upstream inputs are required for a full rerun. This source release does not by itself claim that every analysis can be regenerated from compact summary tables alone.

## Citation and licenses

Citation metadata are in [CITATION.cff](CITATION.cff). Version: 1.0.0. Repository: https://github.com/Zahra-Alavi/ESMfluc.

The authors' code and both trained weight families use MIT. Third-party software and data retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `release/licenses/`.
