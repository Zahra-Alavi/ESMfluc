"""Build and verify a lossless 21-checkpoint Zenodo bundle from completed runs.

Writes only to --output. Source checkpoints and pipeline code are read-only.
"""

from __future__ import annotations

import argparse
import ast
import gc
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from checkpoint_bundle import SCHEMA, sha256_file, tensor_digest


CONDITIONS = (
    "esm2_frozen_linear", "esm2_frozen_bilstm_attn",
    "esm2_top4_bilstm_attn", "esm2_top28_bilstm_attn",
    "esm3_frozen_bilstm_attn", "esm3_top4_bilstm_attn",
    "esm3_top28_bilstm_attn",
)


def parse_training_args(text: str) -> dict:
    """Parse saved argparse Namespace text without evaluating Python code."""
    node = ast.parse(text.strip(), mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) \
            or node.func.id != "Namespace" or node.args:
        raise ValueError("Expected an argparse Namespace")
    result = {}
    for kw in node.keywords:
        if kw.arg is None:
            raise ValueError("Unexpected argument unpacking")
        try:
            result[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, TypeError):
            if kw.arg != "device" or ast.unparse(kw.value) not in (
                "device(type='cuda')", "device(type='cpu')"
            ):
                raise ValueError(f"Unsupported saved argument: {kw.arg}")
            result[kw.arg] = ast.unparse(kw.value)
    return result


def dump_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_tensors(path: Path, tensors: dict):
    """Do not overwrite existing staged weights; support verified resume."""
    if path.exists():
        with safe_open(path, framework="pt", device="cpu") as saved:
            if set(saved.keys()) != set(tensors):
                raise ValueError(f"Existing staged file has different keys: {path}")
            for name, tensor in tensors.items():
                if tensor_digest(saved.get_tensor(name)) != tensor_digest(tensor):
                    raise ValueError(f"Existing staged file has different tensor: {name}")
        return
    temp = path.with_suffix(".partial")
    # Clone only if storage is shared; these checkpoints have independent tensors.
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(temp))
    temp.rename(path)


def build(runs: Path, output: Path, repository: Path, upstream_cache: Path | None = None) -> dict:
    if output.resolve().is_relative_to(runs.resolve()):
        raise ValueError("Output cannot be inside the source run directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": SCHEMA, "release_version": "1.0.0",
                "format": "safetensors shared backbone plus exact tensor overrides",
                "quantization": None, "backbones": {}, "models": []}
    sources = []
    for family in ("esm2", "esm3"):
        reference = runs / f"{family}_frozen_bilstm_attn/seed_1/best_model.pth"
        state = torch.load(reference, map_location="cpu", weights_only=True, mmap=True)
        shared = {k: v for k, v in state.items() if k.startswith("embedding_model.")}
        hashes = {k: tensor_digest(v) for k, v in shared.items()}
        base_name = f"{family}_shared_backbone.safetensors"
        write_tensors(output / base_name, shared)
        manifest["backbones"][family] = {
            "file": base_name, "sha256": sha256_file(output / base_name),
            "size_bytes": (output / base_name).stat().st_size,
            "reference_condition": f"{family}_frozen_bilstm_attn",
            "reference_seed": 1, "tensor_sha256": hashes,
        }
        del state, shared
        gc.collect()
        for condition in (c for c in CONDITIONS if c.startswith(family)):
            for seed in (1, 2, 3):
                print(f"Preparing {condition} seed {seed}", flush=True)
                run = runs / condition / f"seed_{seed}"
                checkpoint = run / "best_model.pth"
                args = parse_training_args((run / "args.txt").read_text())
                if args["seed"] != seed or args["num_classes"] != 2:
                    raise ValueError(f"Unexpected training configuration: {run}")
                state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
                if any(not isinstance(t, torch.Tensor) for t in state.values()):
                    raise ValueError("Expected a tensor-only state_dict")
                tensor_hashes = {k: tensor_digest(v) for k, v in state.items()}
                if not set(hashes).issubset(state):
                    raise ValueError("Backbone tensor names differ from reference")
                overrides = {k: v for k, v in state.items()
                             if tensor_hashes[k] != hashes.get(k)}
                filename = f"{condition}_seed_{seed}.safetensors"
                write_tensors(output / filename, overrides)
                # Check every reconstructed tensor against the original checkpoint.
                with safe_open(output / base_name, framework="pt", device="cpu") as base, \
                        safe_open(output / filename, framework="pt", device="cpu") as delta:
                    if set(base.keys()) | set(delta.keys()) != set(state):
                        raise ValueError("Reconstruction key mismatch")
                    for name, original in state.items():
                        saved = delta if name in overrides else base
                        if tensor_digest(saved.get_tensor(name)) != tensor_hashes[name]:
                            raise ValueError(f"Reconstruction differs at {name}")
                entry = {
                    "condition": condition, "seed": seed, "backbone": family,
                    "file": filename, "sha256": sha256_file(output / filename),
                    "size_bytes": (output / filename).stat().st_size,
                    "source_checkpoint_sha256": sha256_file(checkpoint),
                    "source_checkpoint_size_bytes": checkpoint.stat().st_size,
                    "tensor_order": list(state), "tensor_sha256": tensor_hashes,
                    "state_dict_metadata": getattr(state, "_metadata", None),
                    "training_args": args,
                    "verification": "all tensor names, shapes, dtypes and bytes verified",
                }
                manifest["models"].append(entry)
                sources.append({"condition": condition, "seed": seed,
                                "checkpoint": str(checkpoint.resolve()),
                                "sha256": entry["source_checkpoint_sha256"]})
                config_dir = output / "metadata/models" / condition / f"seed_{seed}"
                config_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(run / "args.txt", config_dir / "args.txt")
                dump_json(config_dir / "training_args.json", args)
                dump_json(output / "manifest.json", manifest)
                print(f"Verified {condition} seed {seed}: "
                      f"{len(overrides)} overrides, {entry['size_bytes'] / 1e9:.3f} GB", flush=True)
                del state, overrides
                gc.collect()
    if len(manifest["models"]) != 21:
        raise ValueError("Expected all seven conditions and all three seeds")
    # Keep private machine paths outside the upload directory.
    dump_json(output.parent / "source_inventory.local.json", sources)
    prepare_metadata(repository, output, manifest, upstream_cache)
    return manifest


def prepare_metadata(repository: Path, output: Path, manifest: dict,
                     upstream_cache: Path | None = None):
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    inputs = ["LICENSE", "pyproject.toml", "requirements.txt", "environment.yaml", "docs/analysis_guide.md",
              "release/zenodo_metadata.draft.json",
              "release/checkpoint_bundle.py", "README.md",
              "release/prepare_zenodo.py", "release/verify_bundle_inference.py",
              "release/download_zenodo.py"]
    inputs += [str(p.relative_to(repository)) for p in (repository / "release/licenses").glob("*")]
    # Archive loader, tokenizers/configs, exact detector and model source files.
    inputs += ["esmfluc/training/models.py", "esmfluc/inference/get_attn.py",
               "esmfluc/analysis/bands/extract_signed_contribution_bands.py",
               "esmfluc/analysis/bands/analyze_signed_band_seed_reproducibility.py"]
    inputs += ["data/atlas_grouped_v1/split_manifest_grouped_v1.csv",
               "data/atlas_grouped_v1/group_manifest_grouped_v1.csv",
               "docs/data_dictionary.md"]
    if (repository / "examples/gk_wt/native_wt.fasta").is_file():
        inputs.append("examples/gk_wt/native_wt.fasta")
    for name in ("gk_wt_contributions_influence.csv", "gk_wt_contributions_bands.csv"):
        candidate = "results/gk_wt_figure/" + name
        if (repository / candidate).is_file():
            inputs.append(candidate)
    for report in output.parent.glob("*_inference_check.local.json"):
        dst = output / "metadata/verification" / report.name.replace(".local", "")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report, dst)
    copied = {}
    for name in inputs:
        src = repository / name
        dst = output / "metadata" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        copied[name] = sha256_file(src)
    manifest["observed_upstream_cache"] = {}
    if upstream_cache is not None:
        for family, cache_name in (
            ("esm2", "models--facebook--esm2_t33_650M_UR50D"),
            ("esm3", "models--EvolutionaryScale--esm3-sm-open-v1"),
        ):
            revision = (upstream_cache / cache_name / "refs/main").read_text().strip()
            snapshot = upstream_cache / cache_name / "snapshots" / revision
            upstream_files = {}
            for filename in ("config.json", "tokenizer_config.json", "special_tokens_map.json",
                             "vocab.txt", "README.md"):
                src = snapshot / filename
                if src.is_file():
                    dst = output / "metadata/upstream" / family / filename
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                    upstream_files[filename] = sha256_file(src)
            manifest["observed_upstream_cache"][family] = {
                "revision": revision, "files": upstream_files,
                "note": "Observed cache revision, not a claim of recorded training-time revision."
            }
    release_metadata = json.loads(
        (repository / "release/zenodo_metadata.draft.json").read_text())["metadata"]
    manifest["licensing"] = {
        "code": "MIT", "weights": {"esm2": "MIT", "esm3": "MIT"},
        "license_file": "LICENSE", "provenance_note": release_metadata["notes"],
    }
    manifest["source_code"] = {"repository": "https://github.com/Zahra-Alavi/ESMfluc",
                               "commit": commit, "file_sha256": copied,
                               "note": "File hashes identify the actual source, including uncommitted release utilities."}
    manifest["prediction"] = {
        "classes": {"0": "single-PB", "1": "multi-PB"},
        "training_target": "mean of three replicate-specific N_eq; single=1, multi>1",
        "maximum_training_length_residues": 1024,
        "default_condition": "esm3_top28_bilstm_attn",
        "ensemble_seeds": [1, 2, 3],
        "ensemble_rule": "Average per-seed probabilities and per-seed contribution matrices, never model weights.",
        "contribution_sign": "positive supports multi-PB; negative supports single-PB",
        "coordinates": "input-sequence positions; map to structure by alignment, not an assumed offset",
        "structure": "optional visualization input, not an ESMfluc prediction input",
    }
    manifest["band_detection"] = {
        "detector_version": "phase1_raw_amplitude_half_intensity_v1",
        "apex_method": "raw_mad_amplitude", "support_method": "half_intensity_merge",
        "amplitude_mad": 2.0, "terminal_exclusion": 0,
        "terminal_exclusion_fraction": 0.0, "smooth": False,
        "half_peak_fraction": 0.5, "seed_matching": "interval_iou",
        "min_interval_iou": 0.5, "minimum_seed_support": 2,
        "profile": "mean of the three per-seed I_j profiles",
        "note": "Use the archived detector for endpoint, zero-MAD and merging behavior."
    }
    manifest["build_environment"] = {name: importlib.metadata.version(name)
                                     for name in ("torch", "safetensors", "numpy")}
    manifest["original_checkpoint_bytes"] = sum(m["source_checkpoint_size_bytes"] for m in manifest["models"])
    manifest["weight_bytes"] = sum(m["size_bytes"] for m in manifest["models"]) + sum(b["size_bytes"] for b in manifest["backbones"].values())
    dump_json(output / "manifest.json", manifest)
    # Bundle metadata separately so a Colab user need download only three seed
    # overrides and one family backbone, rather than a huge combined archive.
    with zipfile.ZipFile(output / "metadata.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted((output / "metadata").rglob("*")):
            if p.is_file():
                archive.write(p, str(p.relative_to(output / "metadata")))
    shutil.copyfile(repository / "release/checkpoint_bundle.py", output / "checkpoint_bundle.py")
    shutil.copyfile(repository / "README.md", output / "README.md")
    shutil.copyfile(repository / "LICENSE", output / "LICENSE")
    files = sorted(p for p in output.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    total = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    print(f"Complete: 21 exact checkpoints; {total / 1e9:.3f} GB staged; "
          f"{len(files) + 1} upload files", flush=True)
    if total > 50_000_000_000:
        raise ValueError("Bundle exceeds the default Zenodo quota; request an increase")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--upstream-cache", type=Path,
                        help="Optional Hugging Face hub cache for small tokenizer/config files")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Refresh metadata/checksums after a completed build")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.metadata_only:
        manifest = json.loads((args.output / "manifest.json").read_text())
        if manifest.get("schema") != SCHEMA or len(manifest.get("models", [])) != 21:
            raise ValueError("Expected a completed 21-checkpoint bundle")
        prepare_metadata(args.repository, args.output, manifest, args.upstream_cache)
    else:
        build(args.runs, args.output, args.repository, args.upstream_cache)


if __name__ == "__main__":
    main()
