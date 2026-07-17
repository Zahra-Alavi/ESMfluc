#!/usr/bin/env python3
"""Plot interactive 3x3 grids of seed-averaged signed logit contributions.

Rows are ESM2 and ESM3. Columns are frozen, top-4, and top-28 backbone
training. Every cell is the elementwise mean of the exact contribution matrix
over all available training seeds for that condition.

For query residue i and key/value residue j:

    C[i, j] > 0  supports flexible (red)
    C[i, j] < 0  supports rigid (blue)

The third row shows non-cancelling key-side flexible and rigid support sums for
both models. Each protein uses one symmetric, zero-centred scale shared by all
six heatmaps. Hover text reports query and key residue number, amino-acid code,
real Neq, the averaged signed contribution, and the training seeds included.
Heatmap hover/click events are linked to a sequence strip and a 3D structure.
"""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from Bio.Align import PairwiseAligner
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.SeqUtils import seq1
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

from plot_attention_structure_grid import (
    PDB_DOWNLOAD_URL,
    find_or_download_structure,
    load_position_map,
    plotly_script_tag,
    position_map_for_protein,
    sequence_plain_html,
    sequence_strip_html,
)

from visualize_exact_logit_contributions import (
    ProteinContribution,
    RunInput,
    choose_proteins,
    common_archive_proteins,
    load_selected_seed,
    resolve_input_path,
    robust_limit,
    safe_filename,
    select_manifest_runs,
)


DEFAULT_MANIFEST = "results/publication_comparable_v2/manifest.tsv"
DEFAULT_TEST_CSV = "data_splits/atlas_grouped_v1/test_grouped_v1.csv"
DEFAULT_OUTPUT_DIR = (
    "results/publication_comparable_v2/averaged_signed_contribution_grids"
)

GRID: Tuple[Tuple[Tuple[str, str], ...], ...] = (
    (
        ("esm2_frozen_bilstm_attn", "Frozen"),
        ("esm2_top4_bilstm_attn", "Top 4"),
        ("esm2_top28_bilstm_attn", "Top 28"),
    ),
    (
        ("esm3_frozen_bilstm_attn", "Frozen"),
        ("esm3_top4_bilstm_attn", "Top 4"),
        ("esm3_top28_bilstm_attn", "Top 28"),
    ),
)
MODEL_BY_ROW = ("ESM2", "ESM3")
# ColorBrewer/Matplotlib RdBu_r: negative is blue, zero is white, positive red.
# Eleven stops closely reproduce the continuous cmap="RdBu_r" used by
# visualize_exact_logit_contributions.py rather than using a coarse 3-stop map.
_RDBU_R = (
    "#053061",
    "#2166ac",
    "#4393c3",
    "#92c5de",
    "#d1e5f0",
    "#f7f7f7",
    "#fddbc7",
    "#f4a582",
    "#d6604d",
    "#b2182b",
    "#67001f",
)
SIGNED_COLORSCALE = tuple(
    (index / (len(_RDBU_R) - 1), color)
    for index, color in enumerate(_RDBU_R)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot ESM2/ESM3 x freezing-condition grids of exact signed "
            "contributions averaged across training seeds."
        )
    )
    parser.add_argument("--manifest_tsv", default=DEFAULT_MANIFEST)
    parser.add_argument("--test_csv", default=DEFAULT_TEST_CSV)
    parser.add_argument(
        "--fasta_file",
        default=None,
        help=(
            "External FASTA providing names/sequences when experimental Neq is "
            "unavailable. Overrides --test_csv; hover reports Neq unavailable."
        ),
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--proteins",
        nargs="*",
        default=None,
        help="Explicit protein names. Default: reproducibly sample proteins.",
    )
    parser.add_argument(
        "--all_proteins",
        action="store_true",
        help="Plot every protein common to all requested runs.",
    )
    parser.add_argument("--n_proteins", type=int, default=5)
    parser.add_argument("--selection_seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Common training seeds to average. Default: 1 2 3.",
    )
    parser.add_argument(
        "--scale_scope",
        choices=("protein", "global"),
        default="protein",
        help="One scale per protein or one scale across every selected protein.",
    )
    parser.add_argument(
        "--scale_quantile",
        type=float,
        default=0.99,
        help="Robust quantile of absolute averaged contributions. Default: 0.99.",
    )
    parser.add_argument("--tick_step", type=int, default=25)
    parser.add_argument("--subplot_size", type=int, default=330)
    parser.add_argument(
        "--include_plotlyjs",
        choices=("cdn", "inline"),
        default="cdn",
        help="Embed Plotly or load it from the CDN. Default: cdn.",
    )
    parser.add_argument(
        "--structure_dir",
        default=None,
        help=(
            "Directory containing/caching PDB files. Default: result-root/pdb_cache "
            "when present, otherwise output_dir/pdb_cache."
        ),
    )
    parser.add_argument(
        "--structure_pattern",
        default="{protein}.pdb",
        help="PDB filename pattern supporting {protein}, {pdb}, and {chain}.",
    )
    parser.add_argument(
        "--download_pdb",
        choices=("auto", "always", "never"),
        default="auto",
    )
    parser.add_argument("--pdb_download_url", default=PDB_DOWNLOAD_URL)
    parser.add_argument("--download_timeout", type=float, default=30.0)
    parser.add_argument(
        "--position_map_csv",
        default=None,
        help="Optional CSV with protein,seq_pos,pdb_chain,pdb_resi columns.",
    )
    parser.add_argument(
        "--allow_structure_sequence_substitutions",
        action="store_true",
        help=(
            "Allow a verified explicit map to place same-length sequence mutants "
            "on a reference PDB template. Positional alignment must remain "
            "consistent; amino-acid substitutions are reported."
        ),
    )
    parser.add_argument(
        "--click_contribution_threshold_fraction",
        type=float,
        default=0.50,
        help=(
            "On heatmap click, highlight query residues whose absolute signed "
            "contribution to the clicked key is at least this fraction of the "
            "shared color limit. Default: 0.50."
        ),
    )
    parser.add_argument(
        "--click_contribution_threshold_abs",
        type=float,
        default=None,
        help="Absolute contribution threshold; overrides the fractional threshold.",
    )
    parser.add_argument("--click_max_query_residues", type=int, default=80)
    parser.add_argument("--reconstruction_atol", type=float, default=1e-5)
    parser.add_argument("--reconstruction_rtol", type=float, default=1e-5)
    args = parser.parse_args()

    if args.n_proteins < 1:
        parser.error("--n_proteins must be positive.")
    if args.all_proteins and args.proteins is not None:
        parser.error("--all_proteins and --proteins are mutually exclusive.")
    if not 0.0 < args.scale_quantile <= 1.0:
        parser.error("--scale_quantile must be in (0, 1].")
    if args.tick_step < 0:
        parser.error("--tick_step cannot be negative.")
    if args.subplot_size < 100:
        parser.error("--subplot_size must be at least 100.")
    if args.click_contribution_threshold_fraction < 0:
        parser.error("--click_contribution_threshold_fraction cannot be negative.")
    if (
        args.click_contribution_threshold_abs is not None
        and args.click_contribution_threshold_abs < 0
    ):
        parser.error("--click_contribution_threshold_abs cannot be negative.")
    if args.click_max_query_residues < 1:
        parser.error("--click_max_query_residues must be positive.")
    if args.seeds is not None and len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds cannot contain duplicates.")
    return args


def parse_neq(value: object) -> np.ndarray:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=float)


def load_fixed_test(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Fixed test CSV does not exist: {path}")
    table = pd.read_csv(path)
    required = {"name", "sequence", "neq"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    result: Dict[str, dict] = {}
    for row in table.itertuples(index=False):
        name = str(row.name)
        if name in result:
            raise ValueError(f"Duplicate protein name in {path}: {name}")
        sequence = str(row.sequence)
        neq = parse_neq(row.neq)
        if neq.shape != (len(sequence),):
            raise ValueError(
                f"{name}: Neq shape {neq.shape} does not match sequence length "
                f"{len(sequence)} in {path}."
            )
        if not np.isfinite(neq).all():
            raise ValueError(f"{name}: real Neq contains non-finite values.")
        result[name] = {"sequence": sequence, "neq": neq}
    if not result:
        raise ValueError(f"No proteins found in {path}.")
    return result


def load_fasta_sequences(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"FASTA does not exist: {path}")
    result: Dict[str, dict] = {}
    name: Optional[str] = None
    chunks: List[str] = []

    def store_record() -> None:
        if name is None:
            return
        sequence = "".join(chunks).replace(" ", "").upper()
        if not sequence:
            raise ValueError(f"{path}: FASTA record {name!r} has an empty sequence.")
        if name in result:
            raise ValueError(f"{path}: duplicate FASTA record name {name!r}.")
        if re.fullmatch(r"[A-Z]+", sequence) is None:
            raise ValueError(f"{path}: FASTA record {name!r} has invalid residues.")
        result[name] = {
            "sequence": sequence,
            "neq": np.full(len(sequence), np.nan, dtype=float),
        }

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            store_record()
            name = line[1:].strip()
            chunks = []
            if not name:
                raise ValueError(f"{path}: FASTA header has no record name.")
        else:
            if name is None:
                raise ValueError(f"{path}: sequence data precedes the first header.")
            chunks.append(line)
    store_record()
    if not result:
        raise ValueError(f"No FASTA records found in {path}.")
    return result


def condition_names() -> List[str]:
    return [condition for row in GRID for condition, _ in row]


def audit_condition_protein_sets(
    runs_by_condition: Mapping[str, Sequence[RunInput]], fixed_test: Mapping[str, dict]
) -> List[str]:
    reference: Optional[set] = None
    reference_condition: Optional[str] = None
    for condition in condition_names():
        names = set(common_archive_proteins(runs_by_condition[condition]))
        if reference is None:
            reference = names
            reference_condition = condition
        elif names != reference:
            raise ValueError(
                f"Protein set for {condition} differs from {reference_condition}; "
                f"missing={sorted(reference - names)[:10]}, "
                f"extra={sorted(names - reference)[:10]}."
            )
    if reference != set(fixed_test):
        raise ValueError(
            "Contribution archives do not match the fixed test split; "
            f"missing={sorted(set(fixed_test) - reference)[:10]}, "
            f"extra={sorted(reference - set(fixed_test))[:10]}."
        )
    return sorted(reference)


def average_condition(
    condition: str,
    runs: Sequence[RunInput],
    proteins: Sequence[str],
    fixed_test: Mapping[str, dict],
    atol: float,
    rtol: float,
) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """Load, validate, align, and elementwise-average all requested seeds."""
    payloads_by_seed: Dict[int, Dict[str, ProteinContribution]] = {}
    for run in runs:
        print(f"Loading {condition}, seed {run.seed} ...")
        payloads_by_seed[run.seed] = load_selected_seed(
            run, proteins, atol=atol, rtol=rtol
        )

    seeds = [run.seed for run in runs]
    averaged: Dict[str, np.ndarray] = {}
    for protein in proteins:
        expected_sequence = str(fixed_test[protein]["sequence"])
        matrices = []
        for seed in seeds:
            payload = payloads_by_seed[seed][protein]
            if payload.sequence != expected_sequence:
                raise ValueError(
                    f"{condition}, seed {seed}, {protein}: sequence does not "
                    "exactly match the fixed test split."
                )
            matrices.append(payload.contribution)
        shapes = {matrix.shape for matrix in matrices}
        if shapes != {(len(expected_sequence), len(expected_sequence))}:
            raise ValueError(
                f"{condition}, {protein}: seed contribution shapes differ: {shapes}"
            )
        # load_selected_seed converts each exact NPZ matrix to float64. Keep
        # that precision and take the literal elementwise arithmetic mean.
        averaged[protein] = np.mean(
            np.stack(matrices, axis=0), axis=0, dtype=np.float64
        )
    return averaged, seeds


def residue_hover_labels(sequence: str, neq: np.ndarray) -> List[str]:
    return [
        (
            f"{index} — {amino_acid} — real Neq {float(neq[index - 1]):.6g}"
            if np.isfinite(neq[index - 1])
            else f"{index} — {amino_acid} — real Neq unavailable"
        )
        for index, amino_acid in enumerate(sequence, start=1)
    ]


def tick_spec(labels: Sequence[str], step: int) -> Tuple[List[str], List[str]]:
    if step == 0:
        return [], []
    positions = list(range(1, len(labels) + 1, step))
    if positions[-1] != len(labels):
        positions.append(len(labels))
    return [labels[position - 1] for position in positions], [
        str(position) for position in positions
    ]


def contribution_limit(matrices: Sequence[np.ndarray], quantile: float) -> float:
    return robust_limit(
        np.concatenate([np.asarray(matrix).ravel() for matrix in matrices]),
        quantile,
    )


def split_protein_id(protein: str) -> Tuple[str, str]:
    parts = str(protein).rsplit("_", 1)
    if len(parts) == 2 and parts[1]:
        return parts[0], parts[1]
    return str(protein), ""


def pdb_chain_residues(
    pdb_path: Path, chain: str
) -> List[Tuple[int, str, str]]:
    """Return unique coordinate residues as (number, insertion code, AA)."""
    residues: List[Tuple[int, str, str]] = []
    seen = set()
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")) or len(line) < 27:
            continue
        if line[21].strip() != chain:
            continue
        try:
            residue_number = int(line[22:26])
        except ValueError:
            continue
        insertion_code = line[26].strip()
        residue_name = line[17:20].strip()
        if line.startswith("HETATM") and residue_name not in protein_letters_3to1_extended:
            continue
        residue_key = (residue_number, insertion_code)
        if residue_key in seen:
            continue
        seen.add(residue_key)
        amino_acid = protein_letters_3to1_extended.get(
            residue_name,
            seq1(
                residue_name,
                custom_map={"MSE": "M", "SEC": "U", "PYL": "O"},
                undef_code="X",
            ),
        )
        residues.append((residue_number, insertion_code, amino_acid))
    return residues


def pdb_seqres_sequence(pdb_path: Path, chain: str) -> str:
    """Read the declared full polymer sequence for one PDB chain."""
    amino_acids: List[str] = []
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("SEQRES") or len(line) < 20:
            continue
        if line[11].strip() != chain:
            continue
        for residue_name in line[19:70].split():
            amino_acids.append(
                protein_letters_3to1_extended.get(
                    residue_name,
                    seq1(
                        residue_name,
                        custom_map={"MSE": "M", "SEC": "U", "PYL": "O"},
                        undef_code="X",
                    ),
                )
            )
    return "".join(amino_acids)


def alignment_position_map(
    protein: str,
    sequence: str,
    pdb_path: Path,
) -> Tuple[Dict[int, Dict[str, str]], str]:
    """Align the matrix sequence to coordinate residues and build hover mapping."""
    _pdb_id, chain = split_protein_id(protein)
    if not chain:
        raise ValueError(
            f"{protein}: cannot infer a PDB chain from the protein identifier. "
            "Provide --position_map_csv."
        )
    residues = pdb_chain_residues(pdb_path, chain)
    if not residues:
        raise ValueError(
            f"{protein}: PDB file {pdb_path} has no ATOM residues for chain "
            f"{chain!r}. Provide --position_map_csv if the chain differs."
        )
    declared_sequence = pdb_seqres_sequence(pdb_path, chain)
    if declared_sequence and declared_sequence != sequence:
        raise ValueError(
            f"{protein}: matrix sequence does not exactly match the PDB SEQRES "
            f"sequence for chain {chain!r} (matrix length={len(sequence)}, "
            f"PDB length={len(declared_sequence)}). Refusing to create a "
            "potentially incorrect structure mapping."
        )
    coordinate_sequence = "".join(amino_acid for _number, _icode, amino_acid in residues)
    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -8.0
    aligner.extend_gap_score = -0.5
    alignments = aligner.align(sequence, coordinate_sequence)
    if len(alignments) != 1:
        raise ValueError(
            f"{protein}: matrix/PDB coordinate alignment is ambiguous "
            f"({len(alignments)} equally scoring alignments). Provide a verified "
            "--position_map_csv."
        )
    alignment = alignments[0]

    position_map: Dict[int, Dict[str, str]] = {}
    identity_matches = 0
    for matrix_block, coordinate_block in zip(
        alignment.aligned[0], alignment.aligned[1]
    ):
        matrix_start, matrix_stop = map(int, matrix_block)
        coordinate_start, coordinate_stop = map(int, coordinate_block)
        if matrix_stop - matrix_start != coordinate_stop - coordinate_start:
            raise AssertionError("Aligned ungapped blocks must have equal lengths.")
        for offset in range(matrix_stop - matrix_start):
            matrix_index = matrix_start + offset
            coordinate_index = coordinate_start + offset
            residue_number, insertion_code, amino_acid = residues[coordinate_index]
            if sequence[matrix_index] == amino_acid:
                identity_matches += 1
            position_map[matrix_index + 1] = {
                "chain": chain,
                "resi": str(residue_number),
                "icode": insertion_code,
            }

    mapped_count = len(position_map)
    identity = identity_matches / mapped_count if mapped_count else 0.0
    if mapped_count == 0 or identity_matches != mapped_count:
        raise ValueError(
            f"{protein}: matrix/PDB chain alignment is unreliable: "
            f"mapped={mapped_count}/{len(sequence)}, identity={identity:.3%}. "
            "Provide a verified --position_map_csv."
        )
    missing_count = len(sequence) - mapped_count
    source = (
        f"sequence_alignment_to_pdb_chain_{chain}; "
        f"SEQRES={'exact' if declared_sequence else 'unavailable'}; "
        f"mapped={mapped_count}/{len(sequence)}; "
        f"identity={identity_matches}/{mapped_count}; "
        f"no_coordinate={missing_count}"
    )
    return position_map, source


_PDB_RESIDUE_ID = re.compile(r"^(-?\d+)([A-Za-z]?)$")


def validate_explicit_position_map(
    protein: str,
    sequence: str,
    pdb_path: Path,
    position_map: Mapping[int, Mapping[str, str]],
    allow_substitutions: bool = False,
) -> Tuple[Dict[int, Dict[str, str]], str]:
    """Require every explicit mapping entry to select the matching PDB amino acid."""
    _pdb_id, inferred_chain = split_protein_id(protein)
    normalized: Dict[int, Dict[str, str]] = {}
    targets = set()
    substitution_count = 0
    residue_cache: Dict[str, Dict[Tuple[int, str], str]] = {}
    for raw_position, raw_entry in position_map.items():
        sequence_position = int(raw_position)
        if not 1 <= sequence_position <= len(sequence):
            raise ValueError(
                f"{protein}: explicit seq_pos {sequence_position} is outside "
                f"1..{len(sequence)}."
            )
        chain = str(raw_entry.get("chain", "")).strip() or inferred_chain
        residue_text = str(raw_entry.get("resi", "")).strip()
        if residue_text.endswith(".0"):
            residue_text = residue_text[:-2]
        match = _PDB_RESIDUE_ID.fullmatch(residue_text)
        if not chain or match is None:
            raise ValueError(
                f"{protein}, seq_pos {sequence_position}: invalid explicit PDB "
                f"target chain={chain!r}, resi={residue_text!r}."
            )
        residue_number = int(match.group(1))
        insertion_code = match.group(2)
        target = (chain, residue_number, insertion_code)
        if target in targets:
            raise ValueError(
                f"{protein}: multiple sequence positions map to PDB target {target}."
            )
        targets.add(target)
        if chain not in residue_cache:
            residue_cache[chain] = {
                (number, icode): amino_acid
                for number, icode, amino_acid in pdb_chain_residues(pdb_path, chain)
            }
        coordinate_amino_acid = residue_cache[chain].get(
            (residue_number, insertion_code)
        )
        sequence_amino_acid = sequence[sequence_position - 1]
        if coordinate_amino_acid is None:
            raise ValueError(
                f"{protein}, seq_pos {sequence_position}: explicit target "
                f"{chain}:{residue_text} has no coordinate residue."
            )
        if coordinate_amino_acid != sequence_amino_acid:
            if not allow_substitutions:
                raise ValueError(
                    f"{protein}, seq_pos {sequence_position}: explicit target "
                    f"{chain}:{residue_text} is {coordinate_amino_acid}, but the "
                    f"matrix sequence is {sequence_amino_acid}."
                )
            substitution_count += 1
        normalized[sequence_position] = {
            "chain": chain,
            "resi": str(residue_number),
            "icode": insertion_code,
        }
    if not normalized:
        raise ValueError(f"{protein}: explicit position map is empty.")
    mapped_chains = {entry["chain"] for entry in normalized.values()}
    if len(mapped_chains) != 1:
        raise ValueError(
            f"{protein}: explicit mapping spans multiple PDB chains "
            f"{sorted(mapped_chains)}; one matrix sequence must map to one chain."
        )
    mapped_chain = next(iter(mapped_chains))
    declared_sequence = pdb_seqres_sequence(pdb_path, mapped_chain)
    if declared_sequence and declared_sequence != sequence:
        if not allow_substitutions or len(declared_sequence) != len(sequence):
            raise ValueError(
                f"{protein}: explicit-map chain {mapped_chain!r} does not have "
                "an exact matrix/PDB SEQRES match."
            )
    residues = pdb_chain_residues(pdb_path, mapped_chain)
    coordinate_sequence = "".join(amino_acid for _n, _i, amino_acid in residues)
    aligner = PairwiseAligner(mode="global")
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -8.0
    aligner.extend_gap_score = -0.5
    candidate_alignments = aligner.align(sequence, coordinate_sequence)
    if len(candidate_alignments) > 10_000:
        raise ValueError(
            f"{protein}: more than 10,000 optimal coordinate alignments; the "
            "explicit mapping cannot be validated safely."
        )
    consistent_candidate_found = False
    for candidate_alignment in candidate_alignments:
        candidate_map: Dict[int, Dict[str, str]] = {}
        for matrix_block, coordinate_block in zip(
            candidate_alignment.aligned[0], candidate_alignment.aligned[1]
        ):
            matrix_start, matrix_stop = map(int, matrix_block)
            coordinate_start, coordinate_stop = map(int, coordinate_block)
            for offset in range(matrix_stop - matrix_start):
                matrix_index = matrix_start + offset
                coordinate_index = coordinate_start + offset
                number, icode, amino_acid = residues[coordinate_index]
                if amino_acid != sequence[matrix_index] and not allow_substitutions:
                    continue
                candidate_map[matrix_index + 1] = {
                    "chain": mapped_chain,
                    "resi": str(number),
                    "icode": icode,
                }
        if all(
            candidate_map.get(position) == target
            for position, target in normalized.items()
        ):
            consistent_candidate_found = True
            break
    if not consistent_candidate_found:
        raise ValueError(
            f"{protein}: explicit targets are not jointly consistent with any "
            "optimal full-chain sequence/coordinate alignment."
        )
    return (
        normalized,
        f"verified_position_map_csv; mapped={len(normalized)}/{len(sequence)}; "
        f"identity={len(normalized) - substitution_count}/{len(normalized)}; "
        f"reference_template_substitutions={substitution_count}",
    )


def figure_for_protein(
    protein: str,
    fixed_entry: Mapping[str, object],
    matrices_by_condition: Mapping[str, Mapping[str, np.ndarray]],
    seeds_by_condition: Mapping[str, Sequence[int]],
    shared_limit: float,
    args: argparse.Namespace,
) -> go.Figure:
    sequence = str(fixed_entry["sequence"])
    neq = np.asarray(fixed_entry["neq"], dtype=float)
    labels = residue_hover_labels(sequence, neq)
    tick_values, tick_text = tick_spec(labels, args.tick_step)
    # Column titles appear once. Model names are separate row labels, avoiding
    # repeated axis/title text between adjacent panels.
    subplot_titles = (
        [title for _condition, title in GRID[0]]
        + ["", "", ""]
        + ["", "", ""]
    )
    figure = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=subplot_titles,
        row_heights=(0.37, 0.37, 0.26),
        horizontal_spacing=0.065,
        vertical_spacing=0.065,
    )

    for row_index, row in enumerate(GRID, start=1):
        for column_index, (condition, freeze_title) in enumerate(row, start=1):
            matrix = matrices_by_condition[condition][protein]
            seeds = seeds_by_condition[condition]
            seeds_text = ", ".join(map(str, seeds))
            panel_title = f"{MODEL_BY_ROW[row_index - 1]} · {freeze_title}"
            figure.add_trace(
                go.Heatmap(
                    z=matrix,
                    x=labels,
                    y=labels,
                    coloraxis="coloraxis",
                    hovertemplate=(
                        f"<b>{panel_title}</b><br>"
                        f"Seeds averaged: {seeds_text}<br>"
                        "Query: %{y}<br>"
                        "Key: %{x}<br>"
                        "Mean signed contribution: %{z:.6g} logit units"
                        "<extra></extra>"
                    ),
                ),
                row=row_index,
                col=column_index,
            )
            figure.update_xaxes(
                tickmode="array",
                tickvals=tick_values,
                ticktext=tick_text,
                showticklabels=False,
                ticks="",
                tickfont=dict(size=10),
                automargin=True,
                row=row_index,
                col=column_index,
            )
            figure.update_yaxes(
                tickmode="array",
                tickvals=tick_values,
                ticktext=tick_text,
                showticklabels=column_index == 1,
                ticks="outside" if column_index == 1 else "",
                tickfont=dict(size=10),
                automargin=True,
                # Plotly categorical heatmaps place the first y category at
                # the bottom by default: query 1 at bottom, query L at top.
                autorange=True,
                row=row_index,
                col=column_index,
            )

    support_max = 0.0
    for column_index in range(1, 4):
        freeze_title = GRID[0][column_index - 1][1]
        for row_index, model in enumerate(MODEL_BY_ROW):
            condition = GRID[row_index][column_index - 1][0]
            matrix = matrices_by_condition[condition][protein]
            flexible_support = np.clip(matrix, 0.0, None).sum(axis=0)
            rigid_support = np.clip(-matrix, 0.0, None).sum(axis=0)
            support_max = max(
                support_max,
                float(np.max(flexible_support)),
                float(np.max(rigid_support)),
            )
            dash = "solid" if model == "ESM2" else "dash"
            for support_name, values, color in (
                ("Flexible", flexible_support, "#b2182b"),
                ("Rigid", rigid_support, "#2166ac"),
            ):
                figure.add_trace(
                    go.Scatter(
                        x=labels,
                        y=values,
                        mode="lines",
                        line=dict(color=color, width=1.6, dash=dash),
                        name=f"{model} {support_name.lower()}",
                        legendgroup=f"{model}_{support_name}",
                        showlegend=column_index == 1,
                        hovertemplate=(
                            f"<b>{model} · {freeze_title} · {support_name} support</b><br>"
                            "Key: %{x}<br>"
                            "Support summed across query residues: %{y:.6g} "
                            "logit units<extra></extra>"
                        ),
                    ),
                    row=3,
                    col=column_index,
                )
        figure.update_xaxes(
            tickmode="array",
            tickvals=tick_values,
            ticktext=tick_text,
            showticklabels=True,
            ticks="outside",
            tickfont=dict(size=10),
            automargin=True,
            row=3,
            col=column_index,
        )
        figure.update_yaxes(
            showticklabels=column_index == 1,
            ticks="outside" if column_index == 1 else "",
            tickfont=dict(size=10),
            rangemode="tozero",
            automargin=True,
            row=3,
            col=column_index,
        )

    profile_upper = support_max * 1.05 if support_max > 0 else 1.0
    for column_index in range(1, 4):
        figure.update_yaxes(range=(0.0, profile_upper), row=3, col=column_index)

    # Shared labels sit outside the subplot domains and cannot overlap an
    # adjacent heatmap. They also make the row/column meaning explicit once.
    figure.add_annotation(
        text="Key/value residue j",
        x=0.5,
        y=-0.10,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=13),
    )
    for label, y_position in (
        ("<b>ESM2</b><br>Query residue i", 0.83),
        ("<b>ESM3</b><br>Query residue i", 0.48),
        ("Key-side support sum", 0.14),
    ):
        figure.add_annotation(
            text=label,
            x=-0.085,
            y=y_position,
            xref="paper",
            yref="paper",
            textangle=-90,
            showarrow=False,
            font=dict(size=13),
        )

    figure.update_layout(
        title=(
            f"{protein}: exact signed flexible-versus-rigid logit contribution"
            "<br><sup>Elementwise mean across training seeds; blue supports rigid, "
            "red supports flexible</sup>"
        ),
        width=args.subplot_size * 3 + 250,
        height=args.subplot_size * 3 + 250,
        coloraxis=dict(
            colorscale=SIGNED_COLORSCALE,
            cmin=-shared_limit,
            cmax=shared_limit,
            cmid=0.0,
            colorbar=dict(
                title="Mean signed<br>contribution<br>(logit units)",
                # Match the combined vertical extent of heatmap rows 1-2.
                # This keeps the bar above the support-profile legend in row 3.
                y=0.6456,
                yanchor="middle",
                len=0.7088,
            ),
        ),
        margin=dict(l=125, r=155, t=125, b=100),
        hoverlabel=dict(align="left"),
        legend=dict(
            x=1.015,
            y=0.22,
            xanchor="left",
            yanchor="middle",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
    )
    return figure


def contribution_viewer_script(
    pdb_text: str,
    position_map: Mapping[int, Mapping[str, str]],
    has_structure: bool,
    shared_limit: float,
    args: argparse.Namespace,
) -> str:
    """Bridge Plotly heatmap hover/click events to sequence and 3D structure."""
    return f"""
<script>
const PDB_TEXT = {json.dumps(pdb_text)};
const POSITION_MAP = {json.dumps(position_map)};
const HAS_STRUCTURE = {json.dumps(bool(has_structure))};
const CONTRIBUTION_ABS_LIMIT = {json.dumps(float(shared_limit))};
const CLICK_THRESHOLD_ABS = {json.dumps(args.click_contribution_threshold_abs)};
const CLICK_THRESHOLD_FRACTION = {json.dumps(args.click_contribution_threshold_fraction)};
const CLICK_MAX_QUERY_RESIDUES = {json.dumps(args.click_max_query_residues)};
let viewer = null;
let lockedSelection = false;

function residueSelection(seqPos) {{
  const entry = POSITION_MAP[String(seqPos)] || POSITION_MAP[seqPos];
  if (!entry) return null;
  const selection = {{}};
  if (entry.chain) selection.chain = entry.chain;
  const resiText = String(entry.resi);
  const resiNum = Number(resiText);
  selection.resi = Number.isFinite(resiNum) && String(resiNum) === resiText ? resiNum : resiText;
  if (entry.icode) selection.icode = String(entry.icode);
  return selection;
}}

function positionFromAxisLabel(label) {{
  const match = String(label).match(/^\s*(\d+)/);
  return match ? Number(match[1]) : null;
}}

function heatmapTrace(point) {{
  const trace = CONTRIBUTION_FIG.data[point.curveNumber];
  return trace && trace.type === "heatmap" ? trace : null;
}}

function pointPositions(point) {{
  const queryPos = positionFromAxisLabel(point.y);
  const keyPos = positionFromAxisLabel(point.x);
  if (!Number.isFinite(queryPos) || !Number.isFinite(keyPos)) return null;
  return {{queryPos: queryPos, keyPos: keyPos}};
}}

function baseStyle() {{
  viewer.setStyle({{}}, {{cartoon: {{color: "green", opacity: 0.82}}}});
}}

function addResidueStyle(selection, color) {{
  if (!selection) return false;
  viewer.setStyle(selection, {{cartoon: {{color: color, opacity: 1.0}}}});
  return true;
}}

function clearSequenceHighlights() {{
  document.querySelectorAll(".seq-residue").forEach(function(el) {{
    el.classList.remove("query-highlight", "key-highlight", "attending-highlight");
  }});
}}

function highlightSequence(queryPos, keyPos) {{
  clearSequenceHighlights();
  const queryEl = document.querySelector('.seq-residue[data-pos="' + queryPos + '"]');
  const keyEl = document.querySelector('.seq-residue[data-pos="' + keyPos + '"]');
  if (queryEl) queryEl.classList.add("query-highlight");
  if (keyEl) {{
    keyEl.classList.remove("query-highlight");
    keyEl.classList.add("key-highlight");
  }}
}}

function highlightSequenceGroup(keyPos, queryPositions, clickedQueryPos) {{
  clearSequenceHighlights();
  queryPositions.forEach(function(pos) {{
    if (Number(pos) === Number(keyPos)) return;
    const el = document.querySelector('.seq-residue[data-pos="' + pos + '"]');
    if (el) el.classList.add("attending-highlight");
  }});
  const queryEl = document.querySelector('.seq-residue[data-pos="' + clickedQueryPos + '"]');
  const keyEl = document.querySelector('.seq-residue[data-pos="' + keyPos + '"]');
  if (queryEl) queryEl.classList.add("query-highlight");
  if (keyEl) {{
    keyEl.classList.remove("query-highlight", "attending-highlight");
    keyEl.classList.add("key-highlight");
  }}
}}

function highlightResidues(queryPos, keyPos) {{
  if (viewer) {{
    baseStyle();
    addResidueStyle(residueSelection(queryPos), "orange");
    addResidueStyle(residueSelection(keyPos), "cyan");
    viewer.render();
  }}
  highlightSequence(queryPos, keyPos);
  const status = document.getElementById("hover-status");
  if (status) {{
    status.textContent = "Query residue " + queryPos + " highlighted orange; key residue " + keyPos + " highlighted cyan.";
  }}
}}

function clickThreshold() {{
  if (CLICK_THRESHOLD_ABS !== null && CLICK_THRESHOLD_ABS !== undefined) {{
    return Number(CLICK_THRESHOLD_ABS);
  }}
  return Number(CLICK_THRESHOLD_FRACTION) * Number(CONTRIBUTION_ABS_LIMIT);
}}

function queryResiduesForClickedKey(point, trace, keyPos) {{
  if (!trace || !trace.z) return [];
  const keyIdx = keyPos - 1;
  const threshold = clickThreshold();
  const rows = [];
  for (let i = 0; i < trace.z.length; i++) {{
    const row = trace.z[i];
    if (!row || keyIdx < 0 || keyIdx >= row.length) continue;
    const value = Number(row[keyIdx]);
    if (Number.isFinite(value) && Math.abs(value) >= threshold) {{
      rows.push({{pos: i + 1, value: value, magnitude: Math.abs(value)}});
    }}
  }}
  rows.sort(function(a, b) {{ return b.magnitude - a.magnitude; }});
  return rows.slice(0, Math.max(1, Number(CLICK_MAX_QUERY_RESIDUES) || rows.length));
}}

function highlightKeyColumn(point) {{
  const trace = heatmapTrace(point);
  const positions = pointPositions(point);
  if (!trace || !positions) return;
  lockedSelection = true;
  let queries = queryResiduesForClickedKey(point, trace, positions.keyPos);
  if (!queries.length) {{
    queries = [{{pos: positions.queryPos, value: Number(point.z), magnitude: Math.abs(Number(point.z))}}];
  }}
  const queryPositions = queries.map(function(item) {{ return item.pos; }});
  if (viewer) {{
    baseStyle();
    queryPositions.forEach(function(pos) {{
      if (pos !== positions.keyPos) addResidueStyle(residueSelection(pos), "#ffd94a");
    }});
    addResidueStyle(residueSelection(positions.queryPos), "orange");
    addResidueStyle(residueSelection(positions.keyPos), "cyan");
    viewer.render();
  }}
  highlightSequenceGroup(positions.keyPos, queryPositions, positions.queryPos);
  const status = document.getElementById("hover-status");
  if (status) {{
    status.textContent = "Clicked key residue " + positions.keyPos + ": highlighted " +
      queryPositions.length + " query residues with |signed contribution| >= " +
      clickThreshold().toPrecision(4) + ". Key is cyan; clicked query is orange; " +
      "other contributing queries are yellow.";
  }}
}}

function initStructureViewer() {{
  const container = document.getElementById("structure-viewer");
  if (!HAS_STRUCTURE || !PDB_TEXT || typeof $3Dmol === "undefined") return;
  viewer = $3Dmol.createViewer(container, {{backgroundColor: "white"}});
  viewer.addModel(PDB_TEXT, "pdb");
  baseStyle();
  viewer.zoomTo();
  viewer.render();
}}

function initHoverBridge() {{
  const plot = document.getElementById("contribution-plot");
  if (!plot || !plot.on) return;
  plot.on("plotly_hover", function(eventData) {{
    if (lockedSelection || !eventData.points || !eventData.points.length) return;
    const point = eventData.points[0];
    if (!heatmapTrace(point)) return;
    const positions = pointPositions(point);
    if (positions) highlightResidues(positions.queryPos, positions.keyPos);
  }});
  plot.on("plotly_click", function(eventData) {{
    if (!eventData.points || !eventData.points.length) return;
    highlightKeyColumn(eventData.points[0]);
  }});
  plot.on("plotly_doubleclick", function() {{
    lockedSelection = false;
    if (viewer) {{
      baseStyle();
      viewer.render();
    }}
    clearSequenceHighlights();
    const status = document.getElementById("hover-status");
    if (status) status.textContent = "Selection cleared. Hover over a contribution heatmap cell to highlight query and key residues.";
  }});
}}

document.addEventListener("DOMContentLoaded", function() {{
  initStructureViewer();
  initHoverBridge();
}});
</script>
"""


def write_protein_page(
    figure: go.Figure,
    protein: str,
    sequence: str,
    output_html: Path,
    pdb_path: Optional[Path],
    pdb_text: str,
    position_map: Mapping[int, Mapping[str, str]],
    map_source: str,
    structure_source: str,
    shared_limit: float,
    args: argparse.Namespace,
) -> None:
    plot_json = json.dumps(figure.to_plotly_json(), cls=PlotlyJSONEncoder)
    has_structure = bool(pdb_text)
    if args.click_contribution_threshold_abs is not None:
        threshold_text = (
            f"|signed contribution| >= {args.click_contribution_threshold_abs:g}"
        )
    else:
        threshold_text = (
            "|signed contribution| >= "
            f"{args.click_contribution_threshold_fraction:g} × shared color limit"
        )
    structure_line = str(pdb_path) if pdb_path else "missing"
    missing_message = ""
    if not has_structure:
        missing_message = (
            "<div class='missing'>No matching PDB structure was found. The contribution "
            "grid and sequence hover remain usable, but linked 3D highlighting is disabled." 
            "</div>"
        )
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(protein)} Signed Contribution Structure Grid</title>",
        plotly_script_tag(args.include_plotlyjs),
        "<script src='https://3Dmol.org/build/3Dmol-min.js'></script>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:0;color:#111;}",
        "header{padding:18px 24px 12px;border-bottom:1px solid #ddd;}",
        "h1{font-size:22px;margin:0 0 8px;}",
        ".meta{color:#444;font-size:13px;line-height:1.45;}",
        ".sequence-panel{position:sticky;top:0;z-index:1000;background:#fff;border-bottom:1px solid #ccc;padding:10px 24px 12px;box-shadow:0 2px 8px rgba(0,0,0,0.08);}",
        ".layout{display:flex;align-items:flex-start;gap:18px;padding:18px 24px 28px;}",
        ".plot-panel{min-width:0;overflow:auto;position:relative;z-index:1;}",
        ".structure-panel{width:420px;min-width:360px;position:sticky;top:16px;z-index:5;background:#fff;}",
        ".sequence-label{font-weight:bold;margin:0 0 6px;}",
        ".sequence-plain{font-family:Consolas,Menlo,monospace;font-size:12px;line-height:1.5;white-space:pre-wrap;word-break:break-all;color:#111!important;border:1px solid #ccc;border-bottom:0;padding:8px;background:#fff;max-height:80px;overflow:auto;margin:0;}",
        "#sequence-strip{font-family:Consolas,Menlo,monospace;font-size:12px;line-height:1.75;word-break:break-all;border:1px solid #ccc;padding:8px;background:#fafafa;max-height:96px;min-height:28px;overflow:auto;color:#111!important;}",
        ".seq-residue{display:inline-block;min-width:1ch;padding:0 2px;border-radius:3px;color:#111!important;}",
        ".seq-residue.query-highlight{background:orange;color:#111;}",
        ".seq-residue.key-highlight{background:cyan;color:#111;}",
        ".seq-residue.attending-highlight{background:#ffe66b;color:#111;}",
        "#structure-viewer{width:100%;height:560px;border:1px solid #ccc;background:#fff;}",
        ".viewer-title{font-weight:bold;margin:0 0 8px;}",
        ".viewer-note,.missing,#hover-status{font-size:13px;line-height:1.45;color:#444;margin-top:8px;}",
        ".missing{padding:12px;border:1px solid #d6a400;background:#fff7d6;color:#4d3a00;}",
        "@media(max-width:1100px){.layout{flex-direction:column}.structure-panel{position:static;width:100%;min-width:0}#structure-viewer{height:480px}}",
        "</style></head><body>",
        "<header>",
        f"<h1>{html.escape(protein)} Signed Contribution Structure Grid</h1>",
        "<div class='meta'>",
        "Rows 1–2: ESM2 and ESM3 seed-averaged signed contribution matrices<br>",
        "Row 3: key-side flexible/rigid support sums; solid=ESM2, dashed=ESM3<br>",
        "Columns: frozen, top4, top28<br>",
        f"Shared signed scale: ±{shared_limit:.6g} logit units<br>",
        f"Structure: {html.escape(structure_line)} ({html.escape(structure_source)})<br>",
        f"Residue mapping: {html.escape(map_source)}<br>",
        f"Click threshold: {html.escape(threshold_text)}",
        "</div></header>",
        "<section class='sequence-panel'>",
        "<div class='sequence-label'>Sequence</div>",
        f"<pre class='sequence-plain'>{sequence_plain_html(sequence)}</pre>",
        f"<div id='sequence-strip'>{sequence_strip_html(sequence)}</div>",
        "</section>",
        "<main class='layout'>",
        "<section class='plot-panel'><div id='contribution-plot'></div></section>",
        "<aside class='structure-panel'>",
        "<div class='viewer-title'>3D structure</div>",
        "<div id='structure-viewer'></div>",
        missing_message,
        "<div id='hover-status'>Hover over a contribution heatmap cell to highlight query and key residues. Click to lock the key column and above-threshold query residues. Double-click to clear.</div>",
        "<div class='viewer-note'>Hover: key/x-axis is cyan, query/y-axis is orange. Click: key is cyan, clicked query is orange, other high-magnitude contributing queries are yellow.</div>",
        "</aside></main>",
        f"<script>const CONTRIBUTION_FIG = {plot_json};",
        "Plotly.newPlot('contribution-plot', CONTRIBUTION_FIG.data, CONTRIBUTION_FIG.layout, {responsive:true,displaylogo:false});</script>",
        contribution_viewer_script(
            pdb_text,
            position_map,
            has_structure,
            shared_limit,
            args,
        ),
        "</body></html>",
    ]
    output_html.write_text("\n".join(parts), encoding="utf-8")


def write_index(
    protein_pages: Sequence[Tuple[str, Path, Optional[Path], str]],
    output_dir: Path,
    summary_path: Path,
) -> Path:
    index_path = output_dir / "index.html"
    links = "\n".join(
        f"<li><a href='{html.escape(page.name)}'>{html.escape(protein)}</a> — "
        f"{'structure found' if pdb_path else 'structure missing'} "
        f"({html.escape(structure_source)})</li>"
        for protein, page, pdb_path, structure_source in protein_pages
    )
    index_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Seed-averaged signed contribution grids</title>"
        "<style>body{font-family:Arial,sans-serif;margin:28px;}"
        "li{margin:9px 0}.note{color:#444;line-height:1.5}</style></head><body>"
        "<h1>Seed-averaged signed contribution grids</h1>"
        "<div class='note'>Rows 1–2: ESM2 and ESM3 signed matrices. Row 3: "
        "key-side support sums. Columns: frozen, top 4, top 28. "
        "Blue supports rigid; red supports flexible. Hover any cell for query and "
        "key residue identity, real Neq, mean contribution, and linked structure highlighting.<br>"
        f"Summary: {html.escape(str(summary_path))}</div><ul>{links}</ul></body></html>",
        encoding="utf-8",
    )
    return index_path


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_tsv).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    if args.fasta_file is not None:
        test_path = resolve_input_path(args.fasta_file, manifest_path)
        fixed_test = load_fasta_sequences(test_path)
        sequence_source_label = "External FASTA (real Neq unavailable)"
    else:
        test_path = resolve_input_path(args.test_csv, manifest_path)
        fixed_test = load_fixed_test(test_path)
        sequence_source_label = "Fixed test CSV"

    runs_by_condition = {
        condition: select_manifest_runs(manifest_path, condition, args.seeds)
        for condition in condition_names()
    }
    available = audit_condition_protein_sets(runs_by_condition, fixed_test)
    proteins, selection_mode = choose_proteins(
        available=available,
        explicit=args.proteins,
        all_proteins=args.all_proteins,
        n_proteins=args.n_proteins,
        selection_seed=args.selection_seed,
    )

    matrices_by_condition: Dict[str, Dict[str, np.ndarray]] = {}
    seeds_by_condition: Dict[str, List[int]] = {}
    for condition in condition_names():
        matrices, seeds = average_condition(
            condition,
            runs_by_condition[condition],
            proteins,
            fixed_test,
            args.reconstruction_atol,
            args.reconstruction_rtol,
        )
        matrices_by_condition[condition] = matrices
        seeds_by_condition[condition] = seeds

    all_selected_matrices = [
        matrices_by_condition[condition][protein]
        for protein in proteins
        for condition in condition_names()
    ]
    global_limit = (
        contribution_limit(all_selected_matrices, args.scale_quantile)
        if args.scale_scope == "global"
        else np.nan
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_proteins.csv"
    with selected_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("selection_rank", "protein", "selection_mode", "selection_seed"))
        for rank, protein in enumerate(proteins, start=1):
            writer.writerow((rank, protein, selection_mode, args.selection_seed))

    position_maps, mapping_summary = load_position_map(args.position_map_csv)
    if args.position_map_csv is None:
        mapping_summary = (
            "No position map CSV provided; deriving matrix-position to PDB-residue "
            "mapping by global sequence alignment within the inferred PDB chain."
        )
    else:
        position_map_table = pd.read_csv(Path(args.position_map_csv).expanduser())
        duplicate_rows = position_map_table.duplicated(
            subset=["protein", "seq_pos"], keep=False
        )
        if duplicate_rows.any():
            examples = position_map_table.loc[
                duplicate_rows, ["protein", "seq_pos"]
            ].head(10)
            raise ValueError(
                "Position map CSV has duplicate (protein, seq_pos) rows: "
                f"{examples.to_dict(orient='records')}"
            )
    if args.structure_dir:
        structure_dir = Path(args.structure_dir).expanduser().resolve()
    else:
        result_pdb_cache = manifest_path.parent / "pdb_cache"
        structure_dir = (
            result_pdb_cache
            if result_pdb_cache.exists()
            else output_dir / "pdb_cache"
        )
    structure_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    protein_pages: List[Tuple[str, Path, Optional[Path], str]] = []
    structure_notes: List[str] = []
    map_notes: List[str] = []
    for protein in proteins:
        protein_matrices = [
            matrices_by_condition[condition][protein]
            for condition in condition_names()
        ]
        limit = (
            global_limit
            if args.scale_scope == "global"
            else contribution_limit(protein_matrices, args.scale_quantile)
        )
        figure = figure_for_protein(
            protein,
            fixed_test[protein],
            matrices_by_condition,
            seeds_by_condition,
            float(limit),
            args,
        )
        sequence = fixed_test[protein]["sequence"]
        pdb_path, candidates, structure_source = find_or_download_structure(
            structure_dir, args.structure_pattern, protein, args
        )
        if protein in position_maps:
            raw_pos_map, _map_source = position_map_for_protein(
                protein, sequence, position_maps
            )
            if pdb_path is None:
                raise ValueError(
                    f"{protein}: cannot validate --position_map_csv because no "
                    "PDB structure was found."
                )
            pos_map, map_source = validate_explicit_position_map(
                protein,
                sequence,
                pdb_path,
                raw_pos_map,
                allow_substitutions=args.allow_structure_sequence_substitutions,
            )
        elif pdb_path is not None:
            pos_map, map_source = alignment_position_map(
                protein, sequence, pdb_path
            )
        else:
            # There is nothing to highlight structurally without a PDB, but
            # retain the attention-grid fallback for a page regenerated later.
            pos_map, map_source = position_map_for_protein(
                protein, sequence, position_maps
            )
        pdb_text = pdb_path.read_text(encoding="utf-8", errors="replace") if pdb_path else ""
        page = output_dir / f"{safe_filename(protein)}_signed_contribution_grid.html"
        write_protein_page(
            figure=figure,
            protein=protein,
            sequence=sequence,
            output_html=page,
            pdb_path=pdb_path,
            pdb_text=pdb_text,
            position_map=pos_map,
            map_source=map_source,
            structure_source=structure_source,
            shared_limit=float(limit),
            args=args,
        )
        protein_pages.append((protein, page, pdb_path, structure_source))
        structure_notes.append(
            f"  {protein}: {pdb_path if pdb_path else 'missing'}; {structure_source} "
            f"(tried: {', '.join(str(candidate) for candidate in candidates)})"
        )
        map_notes.append(f"  {protein}: {map_source}")
        print(f"Wrote {page}")
        for row_index, row in enumerate(GRID):
            for condition, freeze_title in row:
                manifest_rows.append(
                    {
                        "protein": protein,
                        "model": MODEL_BY_ROW[row_index],
                        "freeze_condition": freeze_title,
                        "condition": condition,
                        "seeds_averaged": ",".join(
                            map(str, seeds_by_condition[condition])
                        ),
                        "n_seeds": len(seeds_by_condition[condition]),
                        "sequence_length": len(fixed_test[protein]["sequence"]),
                        "shared_abs_color_limit": float(limit),
                        "structure_path": str(pdb_path) if pdb_path else "",
                        "structure_source": structure_source,
                        "position_map_source": map_source,
                        "output_html": str(page),
                    }
                )

    grid_manifest_path = output_dir / "grid_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(grid_manifest_path, index=False)
    summary_path = output_dir / "summary.txt"
    summary_lines = [
        f"Manifest: {manifest_path}",
        f"Sequence source: {sequence_source_label}: {test_path}",
        f"Selected proteins: {', '.join(proteins)}",
        f"Selection mode: {selection_mode}",
        f"Selection seed: {args.selection_seed}",
        f"Scale scope: {args.scale_scope}",
        f"Scale quantile: {args.scale_quantile}",
        "Rows 1-2: ESM2 and ESM3 signed contribution heatmaps",
        "Row 3: key-side flexible and rigid support sums; solid=ESM2, dashed=ESM3",
        "Columns: frozen, top4, top28",
        "Matrix value: elementwise mean of exact signed contributions across seeds",
        "Key-side flexible support: sum_i max(C[i,j], 0)",
        "Key-side rigid support: sum_i max(-C[i,j], 0)",
        "Color semantics: negative/blue supports rigid; positive/red supports flexible",
        "Hover: query and key residue number, amino-acid code, real Neq, seeds, and mean contribution",
        "Structure interaction: hover highlights query orange and key cyan; click locks the key and high-magnitude query contributors; double-click clears",
        f"Structure dir: {structure_dir}",
        f"Structure pattern: {args.structure_pattern}",
        f"PDB download mode: {args.download_pdb}",
        f"PDB download URL template: {args.pdb_download_url}",
        f"Position mapping: {mapping_summary}",
        f"Reference-template substitutions allowed: {args.allow_structure_sequence_substitutions}",
        f"Click contribution threshold absolute: {args.click_contribution_threshold_abs}",
        f"Click contribution threshold fraction of shared color limit: {args.click_contribution_threshold_fraction}",
        f"Click maximum highlighted query residues: {args.click_max_query_residues}",
        "Default mapping behavior: without --position_map_csv, the matrix sequence is globally aligned to ATOM residues in the PDB chain inferred from the protein-ID suffix. Matrix positions without coordinates remain unmapped.",
        "Structures:",
        *structure_notes,
        "Residue mapping sources:",
        *map_notes,
        "Seeds by condition:",
        *[
            f"  {condition}: {','.join(map(str, seeds_by_condition[condition]))}"
            for condition in condition_names()
        ],
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    index_path = write_index(protein_pages, output_dir, summary_path)
    print(f"Wrote {index_path}")
    print(f"Wrote {grid_manifest_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
