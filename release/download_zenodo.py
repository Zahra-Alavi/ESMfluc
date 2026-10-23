"""Download only one ESMfluc condition from a published Zenodo record.

Uses standard-library Python only. Unpublished drafts are not public download
sources; this helper is for the later Colab workflow after the record is released.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_model(record: int, condition: str, seeds: list[int], destination: Path):
    if record <= 0 or not seeds or any(seed not in (1, 2, 3) for seed in seeds):
        raise ValueError("Expected a positive record ID and seeds 1, 2 or 3")
    base = f"https://zenodo.org/records/{record}/files"
    with urllib.request.urlopen(f"{base}/manifest.json?download=1", timeout=60) as response:
        raw_manifest = response.read(16 * 1024 * 1024)
    manifest = json.loads(raw_manifest)
    if manifest.get("schema") != "esmfluc.checkpoint_bundle.v1":
        raise ValueError("Unsupported model bundle")
    models = []
    for seed in sorted(set(seeds)):
        matches = [m for m in manifest["models"]
                   if m["condition"] == condition and m["seed"] == seed]
        if len(matches) != 1:
            raise ValueError(f"No unique checkpoint for {condition}, seed {seed}")
        models.append(matches[0])
    files = {m["file"]: m["sha256"] for m in models}
    for model in models:
        backbone = manifest["backbones"][model["backbone"]]
        files[backbone["file"]] = backbone["sha256"]
    # The actual loader's source hash is recorded in the manifest.
    files["checkpoint_bundle.py"] = manifest["source_code"]["file_sha256"]["release/checkpoint_bundle.py"]
    if any(Path(name).name != name for name in files):
        raise ValueError("Unsafe filename in remote manifest")
    destination.mkdir(parents=True, exist_ok=True)
    local_manifest = destination / "manifest.json"
    if local_manifest.exists() and json.loads(local_manifest.read_text()) != manifest:
        raise ValueError("Destination already contains a different model manifest")
    local_manifest.write_bytes(raw_manifest)
    for name, expected in files.items():
        path = destination / name
        if path.exists():
            if sha256_file(path) != expected:
                raise ValueError(f"Existing file checksum mismatch: {name}")
            print(f"Already verified: {name}", flush=True)
            continue
        partial = path.with_suffix(path.suffix + ".partial")
        print(f"Downloading {name}", flush=True)
        digest = hashlib.sha256()
        with urllib.request.urlopen(f"{base}/{name}?download=1", timeout=120) as response, \
                partial.open("wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Downloaded file checksum mismatch: {name}")
        partial.rename(path)
    print(f"Verified {condition}, seeds {sorted(set(seeds))}, in {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=int)
    parser.add_argument("--condition", default="esm3_top28_bilstm_attn")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    fetch_model(args.record, args.condition, args.seeds, args.destination)


if __name__ == "__main__":
    main()
