# ESMfluc

ESMfluc predicts whether a residue samples one or several local Protein Block states. Its attention–classifier stage can be decomposed into signed residue contributions, revealing where evidence for each prediction comes from.

## Install

Use Python 3.11 in a new environment, then install from the repository root:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

A GPU is recommended. MMseqs2, ChimeraX and MDStrainMapper are installed separately; see the [analysis guide](docs/analysis_guide.md).

## Predict and explore

The trained models are on [Zenodo](https://doi.org/10.5281/zenodo.23580685). To run ESMfluc in the browser, open the notebook in [Google Colab](https://colab.research.google.com/github/Zahra-Alavi/ESMfluc/blob/main/notebooks/ESMfluc.ipynb). To download and restore models locally:

```bash
python release/download_zenodo.py --record 23580685 --condition esm3_top28_bilstm_attn --seeds 1 2 3 --destination models
python release/checkpoint_bundle.py --bundle models --condition esm3_top28_bilstm_attn --seed 1 --output models/seed_1.pth
```

The [Colab notebook](https://colab.research.google.com/github/Zahra-Alavi/ESMfluc/blob/main/notebooks/ESMfluc.ipynb) takes a sequence and, optionally, a PDB structure. Click a residue to highlight the residues it contributes to. With three seeds, only bands found by at least two seeds are kept; one seed gives candidate bands. Predictions at the first and last two residues lean toward single-PB, because the training labels set N_eq = 1 there by convention.

To run the same viewer locally, also download `metadata.zip` from the Zenodo record into `models`:

```bash
python notebooks/esmfluc_runner.py validate --fasta examples/gk_wt/native_wt.fasta
python notebooks/esmfluc_runner.py predict --bundle models --fasta examples/gk_wt/native_wt.fasta --condition esm3_top28_bilstm_attn --seeds 1 2 3 --device cuda --output runs/guanylate_kinase
python notebooks/esmfluc_runner.py map --output runs/guanylate_kinase --pdb external_inputs/1znx.pdb --chain A
```

The structure is used only for display.

## Train

Reconstruct the fixed ATLAS inputs first, following the analysis guide. Train a condition with:

```bash
esmfluc-train --condition esm2_frozen_bilstm_attn --seed 1 --result_foldername runs/esm2_frozen_bilstm_attn/seed_1
```

Repeat for seeds 1, 2 and 3. The seven conditions are `esm2_frozen_linear` and `{esm2,esm3}_{frozen,top4,top28}_bilstm_attn`.

## Reproduce the paper

[Analysis guide](docs/analysis_guide.md) · [Data dictionary](docs/data_dictionary.md) · [Saved result index](results/README.md) · [Guanylate-kinase example](examples/gk_wt/FIGURE.md).

Result tables are included; trajectories, third-party data and model weights are not. A full rerun needs the Zenodo models and the original ATLAS and mdCATH data.

## Citation and licenses

See [CITATION.cff](CITATION.cff).

Code and trained models are MIT-licensed. Third-party software and data keep their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `release/licenses/`.
