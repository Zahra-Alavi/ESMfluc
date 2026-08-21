#!/usr/bin/env python3
"""Extract 320 K C-alpha RMSF profiles from mdCATH HDF5 trajectories.

For each strict-cohort domain, the five 320 K replicas are independently
least-squares aligned to their first frame on C-alpha atoms. C-alpha RMSF is
computed per replica and then averaged across replicas. Results are checkpointed
after every domain.

Remote mode uses an HTTP range-readable file object, so HDF5 can request the
320 K coordinate chunks without intentionally downloading forces or other
temperatures. The amount transferred can nevertheless be tens to hundreds of
GB; remote execution therefore requires ``--confirm-large-download``.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR / "results" / "benchmark_pegasus_mdcath"
HF_BASE_URL = "https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main/data"

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HSP": "H", "HSD": "H", "HSE": "H", "CYX": "C", "ASH": "D",
    "GLH": "E", "MSE": "M",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Extract strict-cohort mdCATH 320 K C-alpha RMSF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--cohort-csv", type=Path, default=SOURCE_DIR / "mdcath_320K_strict.csv")
    ap.add_argument("--output", type=Path, default=SOURCE_DIR / "mdcath_320K_rmsf.csv")
    ap.add_argument("--failures", type=Path, default=SOURCE_DIR / "mdcath_320K_rmsf_failures.json")
    ap.add_argument("--h5-dir", type=Path, default=None,
                    help="Directory of local mdcath_dataset_<domain>.h5 files; omit for range-readable HTTP")
    ap.add_argument("--base-url", default=HF_BASE_URL)
    ap.add_argument("--temperature", type=int, default=320)
    ap.add_argument("--replicas", default="0,1,2,3,4")
    ap.add_argument("--http-block-mib", type=int, default=8)
    ap.add_argument("--http-max-blocks", type=int, default=32)
    ap.add_argument("--max-domains", type=int, default=None,
                    help="Process only this many unfinished domains (useful for validation)")
    ap.add_argument("--confirm-large-download", action="store_true",
                    help="Required in remote mode because range reads can still transfer a large volume")
    return ap.parse_args()


def parse_vector(value: object) -> list[float]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return [float(v) for v in value]


def parse_pdb_ca(pdb_bytes: bytes) -> tuple[np.ndarray, str]:
    """Return C-alpha atom indices and their one-letter residue sequence."""
    atom_index = -1
    ca_indices, sequence, seen_residues = [], [], set()
    for raw in pdb_bytes.decode("utf-8", errors="replace").splitlines():
        if not raw.startswith(("ATOM  ", "HETATM")):
            continue
        atom_index += 1
        atom_name = raw[12:16].strip()
        altloc = raw[16:17]
        residue_key = (raw[21:22], raw[22:26], raw[26:27])
        if atom_name != "CA" or altloc not in (" ", "A") or residue_key in seen_residues:
            continue
        seen_residues.add(residue_key)
        ca_indices.append(atom_index)
        sequence.append(AA3_TO_1.get(raw[17:20].strip().upper(), "X"))
    if not ca_indices:
        raise ValueError("PDB contains no C-alpha atoms")
    return np.asarray(ca_indices, dtype=int), "".join(sequence)


def coordinate_scale_to_angstrom(first_frame: np.ndarray) -> float:
    """Infer coordinate units from the median adjacent-C-alpha distance."""
    if len(first_frame) < 2:
        return 1.0
    distances = np.linalg.norm(np.diff(first_frame, axis=0), axis=1)
    median = float(np.median(distances[np.isfinite(distances)]))
    if 0.25 <= median <= 0.55:  # nanometers
        return 10.0
    if 2.5 <= median <= 5.5:    # angstroms
        return 1.0
    raise ValueError(f"Unrecognized coordinate units: median adjacent-CA distance={median:g}")


def align_to_reference(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Batch Kabsch alignment for row-vector coordinates [frames, atoms, xyz]."""
    reference_center = reference.mean(axis=0)
    ref0 = reference - reference_center
    centers = mobile.mean(axis=1, keepdims=True)
    mob0 = mobile - centers
    covariance = np.einsum("fai,aj->fij", mob0, ref0, optimize=True)
    u, _, vh = np.linalg.svd(covariance)
    rotation = np.matmul(u, vh)
    reflected = np.linalg.det(rotation) < 0
    if np.any(reflected):
        u[reflected, :, -1] *= -1
        rotation[reflected] = np.matmul(u[reflected], vh[reflected])
    return np.einsum("fai,fij->faj", mob0, rotation, optimize=True) + reference_center


def rmsf_for_replica(coords: np.ndarray) -> np.ndarray:
    reference = coords[0]
    aligned = align_to_reference(coords, reference)
    mean = aligned.mean(axis=0)
    return np.sqrt(np.mean(np.sum((aligned - mean) ** 2, axis=2), axis=0))


class HTTPRangeReader(io.RawIOBase):
    """Seekable block-cached reader backed by HTTP Range requests."""

    def __init__(self, url: str, block_size: int, max_blocks: int):
        super().__init__()
        self.url = url
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.position = 0
        self.session = requests.Session()
        probe = self.session.get(
            url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60,
            allow_redirects=True,
        )
        if probe.status_code != 206:
            probe.close()
            raise OSError(
                f"Server does not support safe byte ranges for {url} "
                f"(HTTP {probe.status_code})"
            )
        content_range = probe.headers.get("Content-Range", "")
        probe.close()
        try:
            self.size = int(content_range.rsplit("/", 1)[1])
        except (IndexError, ValueError) as exc:
            raise OSError(f"Invalid Content-Range for {url}: {content_range!r}") from exc
        self.cache: OrderedDict[int, bytes] = OrderedDict()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = min(position, self.size)
        return self.position

    def _block(self, block_index: int) -> bytes:
        if block_index in self.cache:
            data = self.cache.pop(block_index)
            self.cache[block_index] = data
            return data
        start = block_index * self.block_size
        end = min(start + self.block_size, self.size) - 1
        response = self.session.get(
            self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=180,
            allow_redirects=True,
        )
        if response.status_code != 206:
            response.close()
            raise OSError(
                f"Range request {start}-{end} returned HTTP {response.status_code}"
            )
        data = response.content
        response.close()
        expected = end - start + 1
        if len(data) != expected:
            raise OSError(
                f"Short range response {start}-{end}: got {len(data)}, expected {expected}"
            )
        self.cache[block_index] = data
        while len(self.cache) > self.max_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        pieces = []
        remaining = size
        while remaining:
            block_index = self.position // self.block_size
            within = self.position % self.block_size
            block = self._block(block_index)
            take = min(remaining, len(block) - within)
            pieces.append(block[within:within + take])
            self.position += take
            remaining -= take
        return b"".join(pieces)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self) -> None:
        if not self.closed:
            self.cache.clear()
            self.session.close()
        super().close()


@contextmanager
def open_domain_h5(domain: str, args: argparse.Namespace):
    file_obj = None
    if args.h5_dir is not None:
        path = args.h5_dir / f"mdcath_dataset_{domain}.h5"
        if not path.is_file():
            raise FileNotFoundError(path)
        h5 = h5py.File(path, "r")
    else:
        url = f"{args.base_url.rstrip('/')}/mdcath_dataset_{domain}.h5"
        file_obj = HTTPRangeReader(
            url, block_size=args.http_block_mib * 1024**2,
            max_blocks=args.http_max_blocks,
        )
        h5 = h5py.File(file_obj, "r")
    try:
        yield h5
    finally:
        h5.close()
        if file_obj is not None:
            file_obj.close()


def extract_domain(domain: str, expected_sequence: str, args: argparse.Namespace) -> dict:
    replicas = [int(value) for value in args.replicas.split(",") if value.strip()]
    with open_domain_h5(domain, args) as h5:
        roots = list(h5.keys())
        if len(roots) != 1:
            raise ValueError(f"Expected one HDF5 root group; found {roots}")
        group = h5[roots[0]]
        pdb_data = group["pdbProteinAtoms"][()]
        if isinstance(pdb_data, np.ndarray):
            pdb_data = pdb_data.tobytes()
        ca_indices, pdb_sequence = parse_pdb_ca(pdb_data)
        if pdb_sequence != expected_sequence:
            raise ValueError(
                f"PDB/strict sequence mismatch: pdb={len(pdb_sequence)} strict={len(expected_sequence)}"
            )

        profiles, frame_counts = [], []
        scale = None
        for replica in replicas:
            dataset = group[str(args.temperature)][str(replica)]["coords"]
            coords = np.asarray(dataset[:, ca_indices, :], dtype=np.float64)
            if coords.ndim != 3 or coords.shape[1:] != (len(ca_indices), 3):
                raise ValueError(f"Unexpected coordinate shape for replica {replica}: {coords.shape}")
            replica_scale = coordinate_scale_to_angstrom(coords[0])
            if scale is None:
                scale = replica_scale
            elif replica_scale != scale:
                raise ValueError("Coordinate units differ across replicas")
            profiles.append(rmsf_for_replica(coords * replica_scale))
            frame_counts.append(len(coords))

    return {
        "domain": domain,
        "sequence": expected_sequence,
        "rmsf": np.mean(profiles, axis=0).tolist(),
        "n_replicas": len(profiles),
        "frames_per_replica": frame_counts,
        "temperature_k": args.temperature,
        "units": "angstrom",
        "alignment": "per-replica Kabsch to first frame on all C-alpha atoms",
        "replica_aggregation": "arithmetic mean of per-replica C-alpha RMSF profiles",
    }


def checkpoint(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).sort_values("domain").to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.h5_dir is None and not args.confirm_large_download:
        raise SystemExit(
            "Remote mdCATH RMSF extraction can transfer tens to hundreds of GB. "
            "Re-run with --confirm-large-download, or provide --h5-dir."
        )
    cohort = pd.read_csv(args.cohort_csv, usecols=["domain", "sequence"])
    cohort["domain"] = cohort["domain"].astype(str).str.lstrip(">")
    cohort["sequence"] = cohort["sequence"].astype(str).str.upper()

    rows = []
    if args.output.is_file():
        rows = pd.read_csv(args.output).to_dict("records")
    completed = {str(row["domain"]) for row in rows}
    failures = {}
    if args.failures.is_file():
        failures = json.loads(args.failures.read_text())
    args.failures.parent.mkdir(parents=True, exist_ok=True)

    unfinished = [row for row in cohort.itertuples(index=False) if row.domain not in completed]
    if args.max_domains is not None:
        unfinished = unfinished[:args.max_domains]
    print(f"RMSF: {len(completed)} cached; {len(unfinished)} domains selected")

    for index, row in enumerate(unfinished, start=1):
        print(f"[{index}/{len(unfinished)}] {row.domain}", flush=True)
        try:
            result = extract_domain(row.domain, row.sequence, args)
        except Exception as exc:
            failures[row.domain] = f"{type(exc).__name__}: {exc}"
            args.failures.parent.mkdir(parents=True, exist_ok=True)
            args.failures.write_text(json.dumps(failures, indent=2))
            print(f"  FAILED: {failures[row.domain]}", flush=True)
            continue
        rows.append(result)
        failures.pop(row.domain, None)
        checkpoint(rows, args.output)
        args.failures.write_text(json.dumps(failures, indent=2))
        print(f"  saved {len(result['rmsf'])} residues", flush=True)

    print(f"RMSF complete/cached: {len(rows)}/{len(cohort)}; failures: {len(failures)}")


if __name__ == "__main__":
    main()
