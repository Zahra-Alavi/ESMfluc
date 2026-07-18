#!/usr/bin/env python3
"""Visualize exact flexible-versus-rigid logit contributions.

For a query residue i and key/value residue j, the plotted matrix is

    C[i, j] = A[i, j] * dot(V[j], w_flexible - w_rigid)

Positive values support the flexible class and negative values support the
rigid class.  This script deliberately does not render C as a probability
matrix: the main heatmap uses a diverging colour map centred exactly at zero.

Each selected protein produces one four-row figure with training seeds in
adjacent columns:

  1. signed exact contribution (blue/white/red),
  2. flexible support, max(C, 0),
  3. rigid support, max(-C, 0), and
  4. key-side vertical-band profiles, summed across query residues.

All heatmaps for a protein use one shared contribution-magnitude scale, so
colour intensity is directly comparable across seeds and between flexible and
rigid support.  Unless explicit proteins are supplied, five proteins are
sampled reproducibly from the common protein set using ``--selection_seed``.

The script reads runs from the publication manifest and matches JSON records
to NPZ matrices by unique protein name and the explicit NPZ matrix key.  It
also checks cross-seed protein sets and sequences, shape, residue alignment,
finite values, exact logit-margin reconstruction, and probability consistency
before plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_MANIFEST = "results/publication_comparable_v2/manifest.tsv"
DEFAULT_TEST_CSV = "data_splits/atlas_grouped_v1/test_grouped_v1.csv"
DEFAULT_OUTPUT_DIR = (
    "results/publication_comparable_v2/visualizations_exact_logit_contributions"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot signed exact flexible-versus-rigid logit contributions for "
            "all seeded runs of one BiLSTM-attention condition."
        )
    )
    parser.add_argument("--manifest_tsv", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--test_csv",
        default=DEFAULT_TEST_CSV,
        help="Fixed test split used to audit protein names and sequences.",
    )
    parser.add_argument("--condition", required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Training seeds to compare. Default: every seed for the condition.",
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--protein",
        action="append",
        dest="proteins",
        help="Protein name to plot. Repeat this option to select multiple proteins.",
    )
    selection.add_argument(
        "--all_proteins",
        action="store_true",
        help="Plot every protein shared by the selected seeded runs.",
    )
    parser.add_argument(
        "--n_proteins",
        type=int,
        default=5,
        help="Number of proteins sampled when --protein/--all_proteins is absent.",
    )
    parser.add_argument(
        "--selection_seed",
        type=int,
        default=42,
        help="Random seed for reproducible protein selection. Default: 42.",
    )

    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--robust_quantile",
        type=float,
        default=0.99,
        help="Quantile used for robust heatmap colour limits. Default: 0.99.",
    )
    parser.add_argument(
        "--tick_step",
        type=int,
        default=25,
        help="Residue-position tick interval. Use 0 to hide numbered ticks.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Figure format. Default: png.",
    )
    parser.add_argument(
        "--reconstruction_atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for exact margin reconstruction.",
    )
    parser.add_argument(
        "--reconstruction_rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for exact margin reconstruction.",
    )
    args = parser.parse_args()

    if not 0.0 < args.robust_quantile <= 1.0:
        parser.error("--robust_quantile must be in (0, 1].")
    if args.tick_step < 0:
        parser.error("--tick_step cannot be negative.")
    if args.n_proteins < 1:
        parser.error("--n_proteins must be positive.")
    if args.seeds is not None and len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds cannot contain duplicates.")
    if args.dpi < 1:
        parser.error("--dpi must be positive.")
    if args.reconstruction_atol < 0 or args.reconstruction_rtol < 0:
        parser.error("Reconstruction tolerances cannot be negative.")
    return args


def present(value: object) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def resolve_input_path(raw_path: object, manifest_path: Path) -> Path:
    """Resolve repository-relative manifest paths without changing the CWD."""
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path

    script_dir = Path(__file__).resolve().parent
    candidates = (
        Path.cwd() / path,
        script_dir / path,
        manifest_path.resolve().parent / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


@dataclass(frozen=True)
class RunInput:
    seed: int
    attention_path: Path
    contribution_path: Path


def select_manifest_runs(
    manifest_path: Path, condition: str, requested_seeds: Optional[Sequence[int]]
) -> List[RunInput]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    manifest = pd.read_csv(manifest_path, sep="\t")
    required = {
        "condition",
        "seed",
        "architecture",
        "attention_json",
        "logit_contributions_npz",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(
            f"Manifest {manifest_path} is missing columns: {sorted(missing)}"
        )

    manifest = manifest.copy()
    manifest["seed"] = pd.to_numeric(manifest["seed"], errors="coerce")
    selected = manifest.loc[manifest["condition"].astype(str) == condition].copy()
    if selected.empty:
        raise ValueError(f"No manifest rows found for condition={condition!r}.")
    if (selected["architecture"].astype(str) != "bilstm_attention").any():
        architectures = sorted(selected["architecture"].astype(str).unique())
        raise ValueError(
            "Exact contribution plots require a bilstm_attention condition; "
            f"found architectures={architectures}."
        )
    if selected["seed"].isna().any():
        raise ValueError(f"Condition {condition!r} has a non-numeric seed.")
    selected["seed"] = selected["seed"].astype(int)
    if selected["seed"].duplicated().any():
        duplicates = sorted(selected.loc[selected["seed"].duplicated(), "seed"].tolist())
        raise ValueError(f"Duplicate manifest seeds for {condition!r}: {duplicates}")

    available_seeds = set(selected["seed"].tolist())
    if requested_seeds is not None:
        missing_seeds = sorted(set(requested_seeds) - available_seeds)
        if missing_seeds:
            raise ValueError(
                f"Requested seeds absent for {condition!r}: {missing_seeds}; "
                f"available={sorted(available_seeds)}."
            )
        selected = selected.loc[selected["seed"].isin(requested_seeds)]
    selected = selected.sort_values("seed")

    runs: List[RunInput] = []
    for run in selected.itertuples(index=False):
        if not present(run.logit_contributions_npz):
            raise ValueError(f"Seed {run.seed} has no contribution NPZ in the manifest.")
        attention_path = resolve_input_path(run.attention_json, manifest_path)
        contribution_path = resolve_input_path(
            run.logit_contributions_npz, manifest_path
        )
        for label, path in (
            ("attention JSON", attention_path),
            ("contribution NPZ", contribution_path),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Seed {run.seed}: missing {label}: {path}")
        runs.append(RunInput(int(run.seed), attention_path, contribution_path))
    if len(runs) < 2:
        raise ValueError(
            f"Seed comparison requires at least two runs; found {len(runs)}."
        )
    return runs


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    """Stream objects from a top-level JSON array using only the standard library.

    Attention JSON files are hundreds of MB per run.  Streaming prevents a
    request for one protein from materializing all 208 dense matrices in RAM.
    """
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def read_more() -> None:
            nonlocal buffer, eof
            block = handle.read(chunk_size)
            if block:
                buffer += block
            else:
                eof = True

        def ensure_content() -> None:
            nonlocal buffer, position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                buffer = ""
                position = 0
                read_more()

        read_more()
        ensure_content()
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError(f"Expected a top-level JSON array in {path}")
        position += 1

        while True:
            ensure_content()
            if position < len(buffer) and buffer[position] == "]":
                return

            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"Truncated or invalid JSON array: {path}")
                    buffer = buffer[position:]
                    position = 0
                    read_more()

            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object records in {path}")
            position = end
            ensure_content()
            if position >= len(buffer):
                raise ValueError(f"Truncated JSON array after a record in {path}")
            delimiter = buffer[position]
            if delimiter == ",":
                position += 1
                finished = False
            elif delimiter == "]":
                position += 1
                finished = True
            else:
                raise ValueError(
                    f"Expected ',' or ']' after a record in {path}; got {delimiter!r}"
                )

            # Release the already-decoded dense matrix text before yielding.
            buffer = buffer[position:]
            position = 0
            yield value
            if finished:
                return


def load_archive_index(archive: Mapping[str, np.ndarray], path: Path) -> Dict[str, str]:
    required = {"__protein_names__", "__matrix_keys__"}
    available = set(archive.keys())
    missing = required.difference(available)
    if missing:
        raise ValueError(f"{path} lacks NPZ index arrays: {sorted(missing)}")

    names = np.asarray(archive["__protein_names__"]).astype(str).tolist()
    keys = np.asarray(archive["__matrix_keys__"]).astype(str).tolist()
    if len(names) != len(keys):
        raise ValueError(f"{path}: protein-name and matrix-key counts differ.")
    if len(names) != len(set(names)):
        counts: Dict[str, int] = {}
        for name in names:
            counts[name] = counts.get(name, 0) + 1
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        raise ValueError(f"{path}: duplicate protein names: {duplicates[:10]}")
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate matrix keys in the NPZ index.")

    matrix_arrays = {key for key in available if not key.startswith("__")}
    indexed_arrays = set(keys)
    if matrix_arrays != indexed_arrays:
        missing_arrays = sorted(indexed_arrays - matrix_arrays)
        extra_arrays = sorted(matrix_arrays - indexed_arrays)
        raise ValueError(
            f"{path}: NPZ index/data mismatch; missing={missing_arrays[:10]}, "
            f"unindexed={extra_arrays[:10]}."
        )
    return dict(zip(names, keys))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def validate_record_and_matrix(
    record: Mapping[str, object],
    contribution: np.ndarray,
    expected_key: str,
    contribution_path: Path,
    atol: float,
    rtol: float,
) -> Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "name",
        "sequence",
        "flex_minus_rigid_logit_contribution_file",
        "flex_minus_rigid_logit_contribution_key",
        "flex_minus_rigid_logit_margin",
        "flex_minus_rigid_logit_margin_bias",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"Record is missing fields: {sorted(missing)}")

    name = str(record["name"])
    sequence = str(record["sequence"])
    length = len(sequence)
    stored_key = str(record["flex_minus_rigid_logit_contribution_key"])
    if stored_key != expected_key:
        raise ValueError(
            f"{name}: JSON matrix key {stored_key!r} != indexed key {expected_key!r}."
        )
    stored_file = Path(
        str(record["flex_minus_rigid_logit_contribution_file"])
    ).name
    if stored_file != contribution_path.name:
        raise ValueError(
            f"{name}: JSON contribution file {stored_file!r} does not match "
            f"{contribution_path.name!r}."
        )

    contribution = np.asarray(contribution, dtype=float)
    margin = np.asarray(record["flex_minus_rigid_logit_margin"], dtype=float)
    bias = float(record["flex_minus_rigid_logit_margin_bias"])
    if contribution.shape != (length, length):
        raise ValueError(
            f"{name}: contribution shape {contribution.shape} != {(length, length)}."
        )
    if margin.shape != (length,):
        raise ValueError(f"{name}: margin shape {margin.shape} != {(length,)}.")
    if not np.isfinite(contribution).all() or not np.isfinite(margin).all():
        raise ValueError(f"{name}: contribution or margin contains non-finite values.")
    if not math.isfinite(bias):
        raise ValueError(f"{name}: classifier bias difference is non-finite.")

    reconstructed = contribution.sum(axis=1) + bias
    if not np.allclose(reconstructed, margin, atol=atol, rtol=rtol):
        error = float(np.max(np.abs(reconstructed - margin)))
        raise ValueError(
            f"{name}: exact logit-margin reconstruction failed; max error={error:.3g}."
        )

    if "class_probs" in record:
        class_probs = np.asarray(record["class_probs"], dtype=float)
        if class_probs.shape != (length, 2):
            raise ValueError(
                f"{name}: class_probs shape {class_probs.shape} != {(length, 2)}."
            )
        if not np.allclose(
            sigmoid(margin), class_probs[:, 1], atol=atol, rtol=rtol
        ):
            error = float(np.max(np.abs(sigmoid(margin) - class_probs[:, 1])))
            raise ValueError(
                f"{name}: sigmoid(margin) disagrees with P(flexible); "
                f"max error={error:.3g}."
            )

    flexible = np.clip(contribution, 0.0, None)
    rigid = np.clip(-contribution, 0.0, None)
    return name, sequence, contribution, flexible, rigid


@dataclass
class ProteinContribution:
    sequence: str
    contribution: np.ndarray
    flexible: np.ndarray
    rigid: np.ndarray


def common_archive_proteins(runs: Sequence[RunInput]) -> List[str]:
    """Audit NPZ indexes and require the same protein-name set in every seed."""
    reference: Optional[set] = None
    reference_seed: Optional[int] = None
    for run in runs:
        with np.load(run.contribution_path, allow_pickle=False) as archive:
            names = set(load_archive_index(archive, run.contribution_path))
        if reference is None:
            reference = names
            reference_seed = run.seed
        elif names != reference:
            missing = sorted(reference - names)
            extra = sorted(names - reference)
            raise ValueError(
                f"Seed {run.seed} NPZ protein set differs from seed {reference_seed}; "
                f"missing={missing[:10]}, extra={extra[:10]}."
            )
    if not reference:
        raise ValueError("Contribution archives contain no proteins.")
    return sorted(reference)


def load_fixed_test_sequences(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Fixed test CSV does not exist: {path}")
    table = pd.read_csv(path)
    missing = {"name", "sequence"}.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    names = table["name"].astype(str)
    if names.duplicated().any():
        duplicates = sorted(names[names.duplicated(keep=False)].unique().tolist())
        raise ValueError(f"{path} contains duplicate protein names: {duplicates[:10]}")
    sequences = table["sequence"].astype(str)
    if (sequences.str.len() == 0).any():
        raise ValueError(f"{path} contains an empty sequence.")
    return dict(zip(names, sequences))


def choose_proteins(
    available: Sequence[str],
    explicit: Optional[Sequence[str]],
    all_proteins: bool,
    n_proteins: int,
    selection_seed: int,
) -> Tuple[List[str], str]:
    available_set = set(available)
    if explicit is not None:
        if len(explicit) != len(set(explicit)):
            raise ValueError("A protein was requested more than once.")
        missing = sorted(set(explicit) - available_set)
        if missing:
            raise ValueError(f"Requested proteins are absent: {missing}")
        return list(explicit), "explicit"
    if all_proteins:
        return list(available), "all"
    if n_proteins > len(available):
        raise ValueError(
            f"Cannot sample {n_proteins} proteins from only {len(available)} available."
        )
    rng = np.random.default_rng(selection_seed)
    indices = rng.choice(len(available), size=n_proteins, replace=False)
    return [available[int(index)] for index in indices], "random"


def load_selected_seed(
    run: RunInput,
    selected_names: Sequence[str],
    atol: float,
    rtol: float,
) -> Dict[str, ProteinContribution]:
    """Stream one seed's JSON, validate selected matrices, and audit name sets."""
    selected_set = set(selected_names)
    payloads: Dict[str, ProteinContribution] = {}
    seen_json_names = set()
    with np.load(run.contribution_path, allow_pickle=False) as archive:
        archive_index = load_archive_index(archive, run.contribution_path)
        archive_names = set(archive_index)
        for record in iter_json_array(run.attention_path):
            name = str(record.get("name", ""))
            if not name:
                raise ValueError(
                    f"Seed {run.seed}: attention JSON record has no protein name."
                )
            if name in seen_json_names:
                raise ValueError(
                    f"Seed {run.seed}: duplicate protein name in JSON: {name}"
                )
            seen_json_names.add(name)
            if name not in selected_set:
                continue
            if name not in archive_index:
                raise ValueError(
                    f"Seed {run.seed}, {name}: no name-matched contribution matrix."
                )
            key = archive_index[name]
            validated = validate_record_and_matrix(
                record,
                archive[key],
                key,
                run.contribution_path,
                atol,
                rtol,
            )
            _, sequence, contribution, flexible, rigid = validated
            payloads[name] = ProteinContribution(
                sequence=sequence,
                contribution=contribution,
                flexible=flexible,
                rigid=rigid,
            )

    if seen_json_names != archive_names:
        missing_matrices = sorted(seen_json_names - archive_names)
        orphan_matrices = sorted(archive_names - seen_json_names)
        raise ValueError(
            f"Seed {run.seed}: JSON/NPZ protein-name sets differ; "
            f"JSON without matrices={missing_matrices[:10]}, "
            f"matrices without JSON={orphan_matrices[:10]}."
        )
    missing_selected = sorted(selected_set - set(payloads))
    if missing_selected:
        raise ValueError(
            f"Seed {run.seed}: selected proteins absent from JSON: {missing_selected}"
        )
    return payloads


def robust_limit(values: np.ndarray, quantile: float) -> float:
    """Return a finite, positive robust colour limit."""
    values = np.asarray(values, dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        raise ValueError("Cannot calculate a colour limit from no finite values.")
    limit = float(np.quantile(finite, quantile))
    if limit <= 0:
        limit = float(np.max(finite))
    if limit <= 0:
        # A zero matrix is valid; a unit limit makes its white/empty nature clear.
        limit = 1.0
    return limit


def residue_ticks(length: int, step: int) -> Tuple[np.ndarray, List[str]]:
    if step == 0:
        return np.asarray([], dtype=float), []
    positions = list(range(1, length + 1, step))
    if positions[-1] != length:
        positions.append(length)
    return np.asarray(positions, dtype=float), [str(position) for position in positions]


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "protein"


def plot_seed_comparison(
    name: str,
    sequence: str,
    seed_payloads: Sequence[Tuple[int, ProteinContribution]],
    condition: str,
    output_path: Path,
    robust_quantile: float,
    tick_step: int,
    dpi: int,
) -> List[dict]:
    # Import lazily so validation utilities and tests do not require a display.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    length = len(sequence)
    seeds = [seed for seed, _ in seed_payloads]
    # One absolute contribution scale is shared by every heatmap.  The signed
    # panels use [-limit, +limit], while both magnitude panels use [0, limit].
    combined = np.concatenate(
        [payload.contribution.ravel() for _, payload in seed_payloads]
    )
    magnitude_limit = robust_limit(combined, robust_quantile)

    n_seeds = len(seed_payloads)
    figure, axes = plt.subplots(
        4,
        n_seeds,
        figsize=(5.2 * n_seeds + 1.5, 16.0),
        sharex="col",
        sharey="row",
        constrained_layout=True,
        squeeze=False,
    )
    extent = (0.5, length + 0.5, length + 0.5, 0.5)
    residue_positions = np.arange(1, length + 1)
    ticks, tick_labels = residue_ticks(length, tick_step)
    result_rows: List[dict] = []
    signed_image = flexible_image = rigid_image = None

    for column, (seed, payload) in enumerate(seed_payloads):
        contribution = payload.contribution
        flexible = payload.flexible
        rigid = payload.rigid
        flexible_support = flexible.sum(axis=0)
        rigid_support = rigid.sum(axis=0)
        net_support = contribution.sum(axis=0)

        signed_image = axes[0, column].imshow(
            contribution,
            cmap="RdBu_r",
            vmin=-magnitude_limit,
            vmax=magnitude_limit,
            interpolation="nearest",
            aspect="auto",
            extent=extent,
        )
        flexible_image = axes[1, column].imshow(
            flexible,
            cmap="Reds",
            vmin=0.0,
            vmax=magnitude_limit,
            interpolation="nearest",
            aspect="auto",
            extent=extent,
        )
        rigid_image = axes[2, column].imshow(
            rigid,
            cmap="Blues",
            vmin=0.0,
            vmax=magnitude_limit,
            interpolation="nearest",
            aspect="auto",
            extent=extent,
        )
        axes[3, column].plot(
            residue_positions,
            flexible_support,
            color="#c51b27",
            linewidth=1.2,
            label=r"Flexible  $\sum_i \max(C_{ij},0)$",
        )
        axes[3, column].plot(
            residue_positions,
            rigid_support,
            color="#2166ac",
            linewidth=1.2,
            label=r"Rigid  $\sum_i \max(-C_{ij},0)$",
        )
        axes[3, column].set_xlim(1, length)
        axes[3, column].set_ylim(bottom=0)
        axes[3, column].grid(axis="y", alpha=0.2, linewidth=0.6)
        axes[0, column].set_title(f"Training seed {seed}", fontsize=12)

        for row in range(3):
            axes[row, column].set_xticks(ticks, labels=tick_labels)
            axes[row, column].set_yticks(ticks, labels=tick_labels)
        axes[3, column].set_xticks(ticks, labels=tick_labels)
        axes[3, column].set_xlabel("Key/value residue j (1-based)")

        result_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "protein": name,
                "sequence_length": length,
                "figure": str(output_path),
                "shared_magnitude_limit": magnitude_limit,
                "total_flexible_support": float(flexible_support.sum()),
                "total_rigid_support": float(rigid_support.sum()),
                "total_net_key_support": float(net_support.sum()),
                "max_flexible_support_position": int(
                    np.argmax(flexible_support) + 1
                ),
                "max_rigid_support_position": int(np.argmax(rigid_support) + 1),
            }
        )

    axes[0, 0].set_ylabel("Signed contribution\nQuery residue i")
    axes[1, 0].set_ylabel("Flexible support max(C, 0)\nQuery residue i")
    axes[2, 0].set_ylabel("Rigid support max(−C, 0)\nQuery residue i")
    axes[3, 0].set_ylabel("Key-side support summed\nacross query residues")
    axes[3, 0].legend(frameon=False, fontsize=8, loc="upper right")

    figure.colorbar(
        signed_image,
        ax=list(axes[0, :]),
        shrink=0.75,
        label="Signed contribution (logit units)",
    )
    figure.colorbar(
        flexible_image,
        ax=list(axes[1, :]),
        shrink=0.75,
        label="Flexible-support magnitude (logit units)",
    )
    figure.colorbar(
        rigid_image,
        ax=list(axes[2, :]),
        shrink=0.75,
        label="Rigid-support magnitude (logit units)",
    )

    figure.suptitle(
        f"{name}  |  {condition}  |  seeds {', '.join(map(str, seeds))}  |  L={length}\n"
        f"One shared {100 * robust_quantile:g}% magnitude scale: "
        f"{magnitude_limit:.3g} logit units; "
        "red pushes flexible and blue pushes rigid",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    return result_rows


PROFILE_FIELDS = (
    "condition",
    "seed",
    "protein",
    "residue_position",
    "amino_acid",
    "flexible_support",
    "rigid_support",
    "net_support",
    "absolute_support",
)


def profile_rows(
    condition: str,
    seed: int,
    name: str,
    sequence: str,
    contribution: np.ndarray,
    flexible: np.ndarray,
    rigid: np.ndarray,
) -> Iterable[dict]:
    flexible_support = flexible.sum(axis=0)
    rigid_support = rigid.sum(axis=0)
    net_support = contribution.sum(axis=0)
    for index, amino_acid in enumerate(sequence):
        yield {
            "condition": condition,
            "seed": seed,
            "protein": name,
            "residue_position": index + 1,
            "amino_acid": amino_acid,
            "flexible_support": float(flexible_support[index]),
            "rigid_support": float(rigid_support[index]),
            "net_support": float(net_support[index]),
            "absolute_support": float(
                flexible_support[index] + rigid_support[index]
            ),
        }


def write_visualization_manifest(rows: Sequence[dict], path: Path) -> None:
    fields = (
        "condition",
        "seed",
        "protein",
        "sequence_length",
        "figure",
        "shared_magnitude_limit",
        "total_flexible_support",
        "total_rigid_support",
        "total_net_key_support",
        "max_flexible_support_position",
        "max_rigid_support_position",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_selection_manifest(
    proteins: Sequence[str],
    selection_mode: str,
    selection_seed: int,
    available_count: int,
    seeds: Sequence[int],
    path: Path,
) -> None:
    fields = (
        "selection_rank",
        "protein",
        "selection_mode",
        "selection_seed",
        "available_protein_count",
        "training_seeds",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, protein in enumerate(proteins, start=1):
            writer.writerow(
                {
                    "selection_rank": rank,
                    "protein": protein,
                    "selection_mode": selection_mode,
                    "selection_seed": selection_seed,
                    "available_protein_count": available_count,
                    "training_seeds": ",".join(map(str, seeds)),
                }
            )


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_tsv).expanduser().resolve()
    runs = select_manifest_runs(
        manifest_path, args.condition, args.seeds
    )
    available = common_archive_proteins(runs)
    test_path = resolve_input_path(args.test_csv, manifest_path)
    fixed_test_sequences = load_fixed_test_sequences(test_path)
    if set(available) != set(fixed_test_sequences):
        missing_from_archives = sorted(set(fixed_test_sequences) - set(available))
        extra_in_archives = sorted(set(available) - set(fixed_test_sequences))
        raise ValueError(
            "Contribution archives do not match the fixed test split; "
            f"missing={missing_from_archives[:10]}, extra={extra_in_archives[:10]}."
        )
    selected, selection_mode = choose_proteins(
        available=available,
        explicit=args.proteins,
        all_proteins=args.all_proteins,
        n_proteins=args.n_proteins,
        selection_seed=args.selection_seed,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    if selection_mode == "random":
        selection_directory = (
            f"random_selection_seed_{args.selection_seed}_n_{len(selected)}"
        )
    elif selection_mode == "all":
        selection_directory = "all_proteins"
    else:
        selection_directory = "explicit_proteins"
    run_output_dir = (
        output_dir / safe_filename(args.condition) / selection_directory
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)

    by_seed: Dict[int, Dict[str, ProteinContribution]] = {}
    for run in runs:
        print(f"Loading and validating training seed {run.seed} ...")
        by_seed[run.seed] = load_selected_seed(
            run,
            selected,
            args.reconstruction_atol,
            args.reconstruction_rtol,
        )

    # Cross-seed residue alignment is mandatory: no truncation or padding.
    for protein in selected:
        sequences = {seed: by_seed[seed][protein].sequence for seed in by_seed}
        reference_seed = runs[0].seed
        reference_sequence = sequences[reference_seed]
        mismatched = [
            seed for seed, sequence in sequences.items()
            if sequence != reference_sequence
        ]
        if mismatched:
            raise ValueError(
                f"{protein}: sequence differs across seeds; reference seed "
                f"{reference_seed}, mismatched seeds={mismatched}."
            )
        if reference_sequence != fixed_test_sequences[protein]:
            raise ValueError(
                f"{protein}: seeded-run sequence does not exactly match {test_path}."
            )

    visualized: List[dict] = []
    profile_path = run_output_dir / "contribution_profiles.csv"
    with profile_path.open("w", newline="", encoding="utf-8") as profile_handle:
        profile_writer = csv.DictWriter(profile_handle, fieldnames=PROFILE_FIELDS)
        profile_writer.writeheader()
        for protein in selected:
            sequence = by_seed[runs[0].seed][protein].sequence
            seed_payloads = [
                (run.seed, by_seed[run.seed][protein]) for run in runs
            ]
            figure_path = run_output_dir / f"{safe_filename(protein)}.{args.format}"
            visualized.extend(
                plot_seed_comparison(
                    name=protein,
                    sequence=sequence,
                    seed_payloads=seed_payloads,
                    condition=args.condition,
                    output_path=figure_path,
                    robust_quantile=args.robust_quantile,
                    tick_step=args.tick_step,
                    dpi=args.dpi,
                )
            )
            for seed, payload in seed_payloads:
                profile_writer.writerows(
                    profile_rows(
                        args.condition,
                        seed,
                        protein,
                        sequence,
                        payload.contribution,
                        payload.flexible,
                        payload.rigid,
                    )
                )
            print(f"Wrote {figure_path}")

    manifest_output = run_output_dir / "visualization_manifest.csv"
    write_visualization_manifest(visualized, manifest_output)
    selection_output = run_output_dir / "selected_proteins.csv"
    write_selection_manifest(
        selected,
        selection_mode,
        args.selection_seed,
        len(available),
        [run.seed for run in runs],
        selection_output,
    )
    print(f"Wrote {manifest_output}")
    print(f"Wrote {selection_output}")
    print(f"Wrote {profile_path}")
    print(
        f"Visualized {len(selected)} protein(s) across {len(runs)} training seeds."
    )


if __name__ == "__main__":
    main()
