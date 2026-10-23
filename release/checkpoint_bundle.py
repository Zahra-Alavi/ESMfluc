"""Read losslessly deduplicated ESMfluc model weights; no model downloads.

Requires PyTorch and safetensors. Can also be used as a standalone script.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


SCHEMA = "esmfluc.checkpoint_bundle.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(tensor: torch.Tensor) -> str:
    """Hash dtype, shape and raw bytes, preserving NaNs and signed zero."""
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
    raw = tensor.reshape(-1).view(torch.uint8).numpy()
    digest.update(memoryview(raw))
    return digest.hexdigest()


def local_file(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes bundle: {relative}")
    return path


def read_manifest(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported checkpoint bundle schema")
    return manifest


def model_entry(manifest: dict, condition: str, seed: int) -> dict:
    matches = [m for m in manifest["models"]
               if m["condition"] == condition and m["seed"] == seed]
    if len(matches) != 1:
        raise ValueError(f"Expected one model for {condition}, seed {seed}")
    return matches[0]


def load_bundle_state(root: Path, condition: str, seed: int,
                      verify: bool = True) -> OrderedDict:
    """Reconstruct an original state_dict on CPU, without rounding or deltas.

    Use model.load_state_dict(state, strict=True) after creating the matching
    model. Alternatively export a .pth file for the existing inference CLI.
    Only the selected family's base and the selected seed's override are read.
    """
    root = Path(root)
    manifest = read_manifest(root)
    model = model_entry(manifest, condition, seed)
    base_info = manifest["backbones"][model["backbone"]]
    base = local_file(root, base_info["file"])
    override = local_file(root, model["file"])
    if verify:
        for info, path in ((base_info, base), (model, override)):
            if sha256_file(path) != info["sha256"]:
                raise ValueError(f"Checksum mismatch: {path.name}")
    state = OrderedDict()
    with safe_open(base, framework="pt", device="cpu") as shared, \
            safe_open(override, framework="pt", device="cpu") as specific:
        shared_keys, specific_keys = set(shared.keys()), set(specific.keys())
        expected = set(model["tensor_order"])
        if shared_keys | specific_keys != expected:
            raise ValueError("Bundle tensor names do not match original checkpoint")
        for name in model["tensor_order"]:
            source = specific if name in specific_keys else shared
            tensor = source.get_tensor(name)
            if verify and tensor_digest(tensor) != model["tensor_sha256"][name]:
                raise ValueError(f"Tensor checksum mismatch: {name}")
            state[name] = tensor
    if model.get("state_dict_metadata") is not None:
        state._metadata = OrderedDict(model["state_dict_metadata"].items())
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace {args.output}")
    state = load_bundle_state(args.bundle, args.condition, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    print(f"Restored {len(state)} tensors to {args.output}")


if __name__ == "__main__":
    main()
