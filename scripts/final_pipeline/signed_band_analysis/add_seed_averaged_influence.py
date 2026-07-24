#!/usr/bin/env python3
"""Add the three-seed mean signed column influence to contribution JSON files.

For every condition/split group in the publication manifest, this script:

1. streams the three seed files and verifies identical protein order/sequences;
2. computes mean_seed_I_j = (I_j_seed1 + I_j_seed2 + I_j_seed3) / 3;
3. inserts ``seed_averaged_signed_column_influence`` into every protein record
   in all three seed files; and
4. writes a compact 18-row manifest that points band extraction at one copy of
   each identical averaged profile.

Large gzip files are replaced atomically: the original remains untouched until
the complete rewritten gzip stream closes successfully.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import stat
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .extract_signed_contribution_bands import iter_profiles


AVERAGED_FIELD = "seed_averaged_signed_column_influence"
AVERAGED_DEFINITION_KEY = "seed_averaged_signed_column_influence_definition"


@dataclass(frozen=True)
class ProteinProfile:
    name: str
    sequence: str
    influence: np.ndarray


class StreamCopier:
    """Copy a binary stream while stopping at requested byte markers."""

    def __init__(self, reader, writer, chunk_size: int = 4 * 1024 * 1024):
        self.reader = reader
        self.writer = writer
        self.chunk_size = chunk_size
        self.buffer = b""
        self.eof = False

    def _fill(self) -> bool:
        if self.eof:
            return False
        chunk = self.reader.read(self.chunk_size)
        if not chunk:
            self.eof = True
            return False
        self.buffer += chunk
        return True

    def copy_until(self, marker: bytes) -> None:
        """Copy bytes before marker, then consume (but do not copy) marker."""
        keep = max(0, len(marker) - 1)
        while True:
            index = self.buffer.find(marker)
            if index >= 0:
                self.writer.write(self.buffer[:index])
                self.buffer = self.buffer[index + len(marker):]
                return
            if self.eof:
                raise ValueError(f"Could not find marker {marker!r}")
            if len(self.buffer) > keep:
                cut = len(self.buffer) - keep
                self.writer.write(self.buffer[:cut])
                self.buffer = self.buffer[cut:]
            self._fill()

    def copy_rest(self, forbidden_marker: bytes | None = None) -> None:
        while not self.eof:
            self._fill()
            if forbidden_marker and forbidden_marker in self.buffer:
                raise ValueError(
                    f"Found more protein profiles than expected ({forbidden_marker!r})"
                )
            if self.buffer:
                self.writer.write(self.buffer)
                self.buffer = b""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_tsv", required=True)
    parser.add_argument("--averaged_manifest_tsv", required=True)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--rewrite_workers", type=int, default=3)
    parser.add_argument("--pigz_threads", type=int, default=3)
    parser.add_argument("--compression_level", type=int, default=6)
    return parser.parse_args()


def load_manifest(args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    path = Path(args.manifest_tsv).expanduser().resolve()
    frame = pd.read_csv(path, sep="\t")
    required = {"condition", "seed", "split", "json_gz"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    frame["seed"] = frame["seed"].astype(int)
    if args.conditions:
        frame = frame[frame["condition"].isin(args.conditions)]
    if args.splits:
        frame = frame[frame["split"].isin(args.splits)]
    if frame.empty:
        raise ValueError("No manifest rows selected")
    for (condition, split), group in frame.groupby(["condition", "split"]):
        seeds = sorted(group["seed"].tolist())
        if seeds != [1, 2, 3] or len(group) != 3:
            raise ValueError(f"{condition}/{split} has seeds {seeds}; expected 1,2,3")
        for value in group["json_gz"]:
            if not Path(value).expanduser().is_file():
                raise FileNotFoundError(value)
    return frame.reset_index(drop=True), path


def read_profiles(path: Path) -> list[ProteinProfile]:
    _metadata, profiles = iter_profiles(path)
    return [
        ProteinProfile(
            name=profile["name"],
            sequence=profile["sequence"],
            influence=np.asarray(profile["signed_column_influence"], dtype=np.float64),
        )
        for profile in profiles
    ]


def averaged_payloads(paths_by_seed: dict[int, Path]) -> tuple[list[bytes], int, float]:
    seed_profiles = {seed: read_profiles(path) for seed, path in paths_by_seed.items()}
    reference = seed_profiles[1]
    for seed in (2, 3):
        candidate = seed_profiles[seed]
        if len(candidate) != len(reference):
            raise ValueError(
                f"Seed {seed} has {len(candidate)} proteins; seed 1 has {len(reference)}"
            )
        for index, (left, right) in enumerate(zip(reference, candidate), start=1):
            if left.name != right.name or left.sequence != right.sequence:
                raise ValueError(
                    f"Protein mismatch at row {index}: seed1={left.name}, seed{seed}={right.name}"
                )
            if left.influence.shape != right.influence.shape:
                raise ValueError(f"Influence-length mismatch for {left.name}")

    payloads = []
    residue_count = 0
    max_seed_deviation = 0.0
    for first, second, third in zip(
        seed_profiles[1], seed_profiles[2], seed_profiles[3]
    ):
        stacked = np.stack([first.influence, second.influence, third.influence])
        average = np.mean(stacked, axis=0)
        residue_count += len(average)
        if len(average):
            max_seed_deviation = max(
                max_seed_deviation,
                float(np.max(np.abs(stacked - average[None, :]))),
            )
        payloads.append(
            json.dumps(average.tolist(), separators=(",", ":"), allow_nan=False).encode()
        )
    return payloads, residue_count, max_seed_deviation


def already_contains_average(path: Path) -> bool:
    marker = json.dumps(AVERAGED_DEFINITION_KEY).encode() + b":"
    proteins_marker = b'"proteins":['
    with gzip.open(path, "rb") as handle:
        buffer = b""
        while proteins_marker not in buffer:
            chunk = handle.read(64 * 1024)
            if not chunk:
                break
            buffer += chunk
            if len(buffer) > 8 * 1024 * 1024:
                raise ValueError(f"{path}: proteins header was not found")
    return marker in buffer


def rewrite_one(
    path: Path,
    payloads: list[bytes],
    pigz_threads: int,
    compression_level: int,
) -> tuple[Path, str, int]:
    if already_contains_average(path):
        return path, "already_present", path.stat().st_size

    temporary = path.with_name(f".{path.name}.seed_average_tmp_{os.getpid()}")
    source_mode = stat.S_IMODE(path.stat().st_mode)
    decompressor = None
    compressor = None
    output_handle = None
    try:
        decompressor = subprocess.Popen(
            ["pigz", "-dc", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output_handle = temporary.open("wb")
        compressor = subprocess.Popen(
            ["pigz", f"-{compression_level}", "-p", str(pigz_threads)],
            stdin=subprocess.PIPE,
            stdout=output_handle,
            stderr=subprocess.PIPE,
        )
        assert decompressor.stdout is not None
        assert compressor.stdin is not None
        copier = StreamCopier(decompressor.stdout, compressor.stdin)

        header_marker = b'"protein_count":'
        copier.copy_until(header_marker)
        definition = {
            "shape": "L (key j)",
            "formula": "mean_seed_I_j = (I_j_seed1 + I_j_seed2 + I_j_seed3) / 3",
            "source_seeds": [1, 2, 3],
        }
        compressor.stdin.write(
            json.dumps(AVERAGED_DEFINITION_KEY).encode()
            + b":"
            + json.dumps(definition, separators=(",", ":")).encode()
            + b","
            + header_marker
        )

        influence_marker = b'"signed_column_influence":['
        averaged_key = b',"' + AVERAGED_FIELD.encode() + b'":'
        for payload in payloads:
            copier.copy_until(influence_marker)
            compressor.stdin.write(influence_marker)
            copier.copy_until(b"]")
            compressor.stdin.write(b"]" + averaged_key + payload)
        copier.copy_rest(forbidden_marker=influence_marker)

        compressor.stdin.close()
        compressor_stderr = compressor.stderr.read().decode(errors="replace")
        compressor_rc = compressor.wait()
        decompressor.stdout.close()
        decompressor_stderr = decompressor.stderr.read().decode(errors="replace")
        decompressor_rc = decompressor.wait()
        output_handle.close()
        output_handle = None
        if decompressor_rc != 0:
            raise RuntimeError(f"pigz decompression failed for {path}: {decompressor_stderr}")
        if compressor_rc != 0:
            raise RuntimeError(f"pigz compression failed for {path}: {compressor_stderr}")
        os.chmod(temporary, source_mode)
        os.replace(temporary, path)
        return path, "inserted", path.stat().st_size
    except Exception:
        if compressor is not None and compressor.poll() is None:
            compressor.kill()
        if decompressor is not None and decompressor.poll() is None:
            decompressor.kill()
        if output_handle is not None:
            output_handle.close()
        if temporary.exists():
            temporary.unlink()
        raise


def atomic_write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp_{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.rewrite_workers < 1 or args.pigz_threads < 1:
        raise ValueError("Worker and pigz thread counts must be positive")
    if not 1 <= args.compression_level <= 9:
        raise ValueError("--compression_level must be between 1 and 9")
    if shutil.which("pigz") is None:
        raise RuntimeError("pigz is required for streaming gzip rewrite")

    selected, original_manifest_path = load_manifest(args)
    audit_rows = []
    groups = list(selected.groupby(["condition", "split"], sort=False))
    for group_number, ((condition, split), group) in enumerate(groups, start=1):
        paths_by_seed = {
            int(row.seed): Path(row.json_gz).expanduser().resolve()
            for row in group.itertuples(index=False)
        }
        print(
            f"[{group_number}/{len(groups)}] {condition}/{split}: reading three seeds",
            flush=True,
        )
        payloads, residue_count, max_seed_deviation = averaged_payloads(paths_by_seed)
        print(
            f"[{group_number}/{len(groups)}] {condition}/{split}: "
            f"{len(payloads)} proteins, {residue_count} residues; rewriting files",
            flush=True,
        )
        futures = {}
        with ThreadPoolExecutor(max_workers=min(args.rewrite_workers, 3)) as executor:
            for seed, path in paths_by_seed.items():
                future = executor.submit(
                    rewrite_one,
                    path,
                    payloads,
                    args.pigz_threads,
                    args.compression_level,
                )
                futures[future] = seed
            for future in as_completed(futures):
                seed = futures[future]
                path, status, compressed_bytes = future.result()
                print(
                    f"[{group_number}/{len(groups)}] {condition}/{split} "
                    f"seed={seed}: {status} ({compressed_bytes / 1024**3:.2f} GiB)",
                    flush=True,
                )
                audit_rows.append({
                    "condition": condition,
                    "split": split,
                    "seed": seed,
                    "json_gz": str(path),
                    "protein_count": len(payloads),
                    "residue_count": residue_count,
                    "max_abs_seed_deviation_from_mean_I_j": max_seed_deviation,
                    "status": status,
                    "compressed_bytes": compressed_bytes,
                })

    # Refresh sizes and document the new field in the original manifest.
    full_manifest = pd.read_csv(original_manifest_path, sep="\t")
    full_manifest["compressed_bytes"] = [
        Path(value).expanduser().stat().st_size for value in full_manifest["json_gz"]
    ]
    full_manifest["seed_averaged_I_j_field"] = AVERAGED_FIELD
    full_manifest["seed_averaged_from_seeds"] = "1,2,3"
    atomic_write_tsv(full_manifest, original_manifest_path)

    # One file per condition/split is sufficient because all three copies of
    # the averaged profile are identical.
    averaged_rows = []
    for (condition, split), group in selected.groupby(["condition", "split"], sort=False):
        representative = group.sort_values("seed").iloc[0]
        averaged_rows.append({
            "condition": condition,
            "seed": "average",
            "split": split,
            "json_gz": str(Path(representative.json_gz).expanduser().resolve()),
            "source_seeds": "1,2,3",
            "influence_field": AVERAGED_FIELD,
        })
    averaged_manifest_path = Path(args.averaged_manifest_tsv).expanduser().resolve()
    atomic_write_tsv(pd.DataFrame(averaged_rows), averaged_manifest_path)
    audit_path = averaged_manifest_path.with_name("seed_averaged_influence_audit.tsv")
    atomic_write_tsv(pd.DataFrame(audit_rows), audit_path)
    print(json.dumps({
        "files_processed": len(audit_rows),
        "averaged_manifest_tsv": str(averaged_manifest_path),
        "audit_tsv": str(audit_path),
        "field": AVERAGED_FIELD,
    }, indent=2))


if __name__ == "__main__":
    main()
