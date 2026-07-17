#!/usr/bin/env python3
"""Extract exact signed contribution matrices for an external FASTA.

The script re-runs Attention/get_attn.py for the six publication BiLSTM
conditions and requested training seeds, validates every emitted matrix, and
writes a manifest consumable by plot_averaged_signed_contribution_grid.py.
Runs are resumable and may be distributed across multiple GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


CONDITIONS = (
    "esm2_frozen_bilstm_attn",
    "esm2_top4_bilstm_attn",
    "esm2_top28_bilstm_attn",
    "esm3_frozen_bilstm_attn",
    "esm3_top4_bilstm_attn",
    "esm3_top28_bilstm_attn",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta_file", required=True)
    parser.add_argument(
        "--source_manifest",
        default="results/publication_comparable_v2/manifest.tsv",
    )
    parser.add_argument(
        "--output_root",
        default="results/weinreb2025_mutants_no_AV/exact_contributions_v2",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--gpus", nargs="+", default=["0", "1"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--get_attn_script", default="Attention/get_attn.py"
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive.")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds cannot contain duplicates.")
    unknown = sorted(set(args.conditions) - set(CONDITIONS))
    if unknown:
        parser.error(f"Unsupported conditions: {unknown}")
    return args


def parse_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    name = None
    chunks: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(chunks).upper()))
            name = line[1:].strip()
            chunks = []
        else:
            if name is None:
                raise ValueError("FASTA sequence appears before the first header.")
            chunks.append(line)
    if name is not None:
        records.append((name, "".join(chunks).upper()))
    names = [name for name, _sequence in records]
    if not records or len(names) != len(set(names)):
        raise ValueError("FASTA is empty or has duplicate record names.")
    if any(not sequence for _name, sequence in records):
        raise ValueError("FASTA contains an empty sequence.")
    return records


def resolve_path(value: object, manifest_path: Path) -> Path:
    raw = Path(str(value)).expanduser()
    candidates = (
        [raw]
        if raw.is_absolute()
        else [Path.cwd() / raw, manifest_path.parent / raw, raw]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def validate_outputs(
    attention_path: Path,
    contribution_path: Path,
    fasta_records: Sequence[Tuple[str, str]],
) -> Dict[str, float]:
    if not attention_path.is_file() or not contribution_path.is_file():
        raise FileNotFoundError("Attention JSON or contribution NPZ is missing.")
    records = json.loads(attention_path.read_text(encoding="utf-8"))
    expected = dict(fasta_records)
    if [str(record.get("name")) for record in records] != list(expected):
        raise ValueError("Attention JSON names/order do not match the FASTA.")
    max_reconstruction_error = 0.0
    max_probability_error = 0.0
    with np.load(contribution_path, allow_pickle=False) as archive:
        names = archive["__protein_names__"].astype(str).tolist()
        keys = archive["__matrix_keys__"].astype(str).tolist()
        if names != list(expected) or len(keys) != len(set(keys)):
            raise ValueError("Contribution NPZ index does not match the FASTA.")
        lookup = dict(zip(names, keys))
        for record in records:
            name = str(record["name"])
            sequence = str(record["sequence"])
            if sequence != expected[name]:
                raise ValueError(f"{name}: JSON sequence differs from FASTA.")
            if record.get("flex_minus_rigid_logit_contribution_file") != contribution_path.name:
                raise ValueError(
                    f"{name}: JSON contribution filename does not match the NPZ."
                )
            matrix = np.asarray(archive[lookup[name]], dtype=np.float64)
            length = len(sequence)
            if matrix.shape != (length, length) or not np.isfinite(matrix).all():
                raise ValueError(f"{name}: invalid contribution matrix.")
            margin = np.asarray(
                record["flex_minus_rigid_logit_margin"], dtype=np.float64
            )
            bias = float(record["flex_minus_rigid_logit_margin_bias"])
            reconstruction_error = float(
                np.max(np.abs(matrix.sum(axis=1) + bias - margin))
            )
            max_reconstruction_error = max(
                max_reconstruction_error, reconstruction_error
            )
            probabilities = np.asarray(record["class_probs"], dtype=np.float64)
            probability_error = float(
                np.max(np.abs(sigmoid(margin) - probabilities[:, 1]))
            )
            max_probability_error = max(max_probability_error, probability_error)
    if max_reconstruction_error > 1e-5 or max_probability_error > 1e-5:
        raise ValueError(
            "Exact-contribution validation failed: "
            f"reconstruction={max_reconstruction_error:.3g}, "
            f"probability={max_probability_error:.3g}."
        )
    return {
        "max_reconstruction_error": max_reconstruction_error,
        "max_probability_error": max_probability_error,
    }


def select_runs(
    manifest_path: Path, conditions: Sequence[str], seeds: Sequence[int]
) -> List[dict]:
    table = pd.read_csv(manifest_path, sep="\t")
    selected = table[
        table["condition"].isin(conditions)
        & table["seed"].astype(int).isin(seeds)
    ].copy()
    selected["seed"] = selected["seed"].astype(int)
    expected = {(condition, seed) for condition in conditions for seed in seeds}
    observed = set(zip(selected["condition"], selected["seed"]))
    if observed != expected or selected.duplicated(["condition", "seed"]).any():
        raise ValueError(
            f"Source manifest run mismatch; missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}."
        )
    if (selected["architecture"].astype(str) != "bilstm_attention").any():
        raise ValueError("All selected runs must use bilstm_attention.")
    rows = []
    condition_order = {condition: index for index, condition in enumerate(conditions)}
    selected["condition_order"] = selected["condition"].map(condition_order)
    for row in selected.sort_values(["condition_order", "seed"]).to_dict("records"):
        row["checkpoint_path"] = resolve_path(row["checkpoint"], manifest_path)
        if not row["checkpoint_path"].is_file():
            raise FileNotFoundError(row["checkpoint_path"])
        rows.append(row)
    return rows


def run_one(
    row: dict,
    task_index: int,
    args: argparse.Namespace,
    fasta_path: Path,
    fasta_records: Sequence[Tuple[str, str]],
    output_root: Path,
    get_attn_path: Path,
) -> dict:
    condition = str(row["condition"])
    seed = int(row["seed"])
    gpu = str(args.gpus[task_index % len(args.gpus)])
    run_dir = output_root / "runs" / condition / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    attention_path = run_dir / "attention.json"
    contribution_path = run_dir / "flex_rigid_logit_contributions.npz"
    log_path = run_dir / "extraction.log"

    if not args.force:
        try:
            audit = validate_outputs(attention_path, contribution_path, fasta_records)
            return {**row, **audit, "status": "reused", "gpu": gpu,
                    "run_dir_external": run_dir,
                    "attention_external": attention_path,
                    "contribution_external": contribution_path}
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            pass

    partial_dir = run_dir / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_attention = partial_dir / "attention.json"
    partial_contribution = partial_dir / "flex_rigid_logit_contributions.npz"
    command = [
        sys.executable,
        str(get_attn_path),
        "--checkpoint", str(row["checkpoint_path"]),
        "--fasta_file", str(fasta_path),
        "--architecture", "bilstm_attention",
        "--esm_model", str(row["esm_model"]),
        "--hidden_size", "512",
        "--num_layers", "3",
        "--dropout", "0.0",
        "--bidirectional", "1",
        "--num_classes", "2",
        "--output", str(partial_attention),
        "--logit_contributions_output", str(partial_contribution),
    ]
    if bool(row["is_esm3"]):
        command.append("--is_esm3")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            print(f"[{condition} seed={seed} gpu={gpu}] {line}", end="", flush=True)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"{condition} seed {seed} failed with exit code {return_code}; "
            f"see {log_path}"
        )
    audit = validate_outputs(partial_attention, partial_contribution, fasta_records)
    os.replace(partial_attention, attention_path)
    os.replace(partial_contribution, contribution_path)
    return {**row, **audit, "status": "extracted", "gpu": gpu,
            "run_dir_external": run_dir,
            "attention_external": attention_path,
            "contribution_external": contribution_path}


def write_manifest(results: Sequence[dict], output_root: Path) -> Path:
    rows = []
    for result in sorted(results, key=lambda item: (CONDITIONS.index(item["condition"]), item["seed"])):
        rows.append({
            "condition": result["condition"],
            "seed": result["seed"],
            "architecture": "bilstm_attention",
            "esm_model": result["esm_model"],
            "is_esm3": bool(result["is_esm3"]),
            "freeze_mode": result.get("freeze_mode", ""),
            "freeze_layers": result.get("freeze_layers", ""),
            "run_dir": str(Path(result["run_dir_external"]).resolve()),
            "attention_json": str(Path(result["attention_external"]).resolve()),
            "logit_contributions_npz": str(Path(result["contribution_external"]).resolve()),
            "backbone_attention_json": "",
            "checkpoint": str(Path(result["checkpoint_path"]).resolve()),
        })
    path = output_root / "manifest.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def run_gpu_queue(
    gpu: str,
    queued_runs: Sequence[Tuple[int, dict]],
    args: argparse.Namespace,
    fasta_path: Path,
    fasta_records: Sequence[Tuple[str, str]],
    output_root: Path,
    get_attn_path: Path,
) -> Tuple[List[dict], List[Tuple[str, int, str]]]:
    """Run one strictly sequential queue on one GPU."""
    gpu_args = argparse.Namespace(**vars(args))
    gpu_args.gpus = [gpu]
    completed: List[dict] = []
    failures: List[Tuple[str, int, str]] = []
    for _original_index, row in queued_runs:
        condition = str(row["condition"])
        seed = int(row["seed"])
        try:
            result = run_one(
                row,
                0,
                gpu_args,
                fasta_path,
                fasta_records,
                output_root,
                get_attn_path,
            )
            completed.append(result)
            print(f"COMPLETE {condition} seed={seed}: {result['status']}", flush=True)
        except Exception as exc:  # keep the GPU queue moving after one failure
            failures.append((condition, seed, str(exc)))
            print(f"FAILED {condition} seed={seed}: {exc}", file=sys.stderr, flush=True)
    return completed, failures


def main() -> None:
    args = parse_args()
    fasta_path = Path(args.fasta_file).expanduser().resolve()
    manifest_path = Path(args.source_manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    get_attn_path = Path(args.get_attn_script).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    fasta_records = parse_fasta(fasta_path)
    runs = select_runs(manifest_path, args.conditions, args.seeds)
    print(
        f"Extracting {len(runs)} collections for {len(fasta_records)} FASTA "
        f"records with workers={args.workers}, GPUs={args.gpus}",
        flush=True,
    )
    results: List[dict] = []
    failures: List[Tuple[str, int, str]] = []
    active_gpus = list(args.gpus[: min(args.workers, len(args.gpus))])
    queues: Dict[str, List[Tuple[int, dict]]] = {gpu: [] for gpu in active_gpus}
    for index, row in enumerate(runs):
        queues[active_gpus[index % len(active_gpus)]].append((index, row))
    with ThreadPoolExecutor(max_workers=len(active_gpus)) as executor:
        futures = {
            executor.submit(
                run_gpu_queue,
                gpu,
                queue,
                args,
                fasta_path,
                fasta_records,
                output_root,
                get_attn_path,
            ): gpu
            for gpu, queue in queues.items()
        }
        for future in as_completed(futures):
            queue_results, queue_failures = future.result()
            results.extend(queue_results)
            failures.extend(queue_failures)
    manifest = write_manifest(results, output_root)
    summary = {
        "fasta": str(fasta_path),
        "source_manifest": str(manifest_path),
        "output_manifest": str(manifest),
        "records": len(fasta_records),
        "completed": len(results),
        "failures": failures,
        "runs": [
            {
                "condition": result["condition"],
                "seed": int(result["seed"]),
                "status": result["status"],
                "gpu": result["gpu"],
                "max_reconstruction_error": result["max_reconstruction_error"],
                "max_probability_error": result["max_probability_error"],
            }
            for result in results
        ],
    }
    (output_root / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
