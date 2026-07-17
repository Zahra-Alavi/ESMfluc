#!/usr/bin/env python3
"""Extract exact signed evidence and attention for every publication-v2 split.

One model checkpoint is loaded once, then each requested split is streamed to a
gzip-compressed JSON document.  Each protein record contains exactly the five
requested model-analysis arrays plus identifying and reconstruction metadata:

  contribution_matrix       C_ij = A_ij * s_j                    [L, L]
  intrinsic_signed_evidence s_j                                  [L]
  signed_column_influence   I_j = (1/L) sum_i C_ij               [L]
  attention_matrix          A_ij                                 [L, L]
  attention_column_mean     (1/L) sum_i A_ij                     [L]

Positive values support flexible (class 1, Neq > 1); negative values support
rigid (class 0, Neq <= 1).  Files are written atomically, so a final .json.gz
file is evidence that all requested records were serialized successfully.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import EsmModel, EsmTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from get_attn import (  # noqa: E402
    ESM3_AVAILABLE,
    ESM3Wrapper,
    EsmSequenceTokenizer,
    ESM3_sm_open_v0,
    infer_bilstm_params,
    run_model_bilstm_attn,
)
from models import BiLSTMWithSelfAttentionModel  # noqa: E402


SCHEMA_VERSION = "esmfluc.signed_contributions.v1"
ARRAY_DEFINITIONS = {
    "contribution_matrix": {
        "shape": "L x L (query i, key j)",
        "formula": "C_ij = A_ij * s_j",
    },
    "intrinsic_signed_evidence": {
        "shape": "L (key j)",
        "formula": "s_j = dot(V_j, fc.weight[1] - fc.weight[0])",
    },
    "signed_column_influence": {
        "shape": "L (key j)",
        "formula": "I_j = (1/L) * sum_i C_ij",
    },
    "attention_matrix": {
        "shape": "L x L (query i, key j)",
        "formula": "A_ij",
    },
    "attention_column_mean": {
        "shape": "L (key j)",
        "formula": "(1/L) * sum_i A_ij",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--esm_model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--is_esm3", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", type=int, default=1)
    parser.add_argument("--split", action="append", nargs=2, metavar=("NAME", "CSV"), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--compression_level", type=int, default=4, choices=range(1, 10))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_proteins", type=int, default=None,
                        help="Validation/debugging only; omit for complete extraction.")
    return parser.parse_args()


def load_split(path: Path, max_proteins: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"name", "sequence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
    if frame["name"].duplicated().any():
        duplicates = frame.loc[frame["name"].duplicated(), "name"].head().tolist()
        raise ValueError(f"{path} has duplicate protein names: {duplicates}")
    frame = frame.loc[:, ["name", "sequence"]].copy()
    frame["name"] = frame["name"].astype(str)
    frame["sequence"] = frame["sequence"].astype(str)
    lengths = frame["sequence"].str.len()
    if (lengths <= 0).any():
        raise ValueError(f"{path} contains an empty sequence")
    if (lengths > 1024).any():
        offenders = frame.loc[lengths > 1024, "name"].tolist()
        raise ValueError(f"{path} contains sequences longer than 1024: {offenders[:10]}")
    if max_proteins is not None:
        frame = frame.head(max_proteins).copy()
    return frame


def load_model(args: argparse.Namespace, device: torch.device):
    if args.is_esm3:
        if not ESM3_AVAILABLE:
            raise ImportError("ESM3 is unavailable in the active Python environment")
        raw_backbone = ESM3_sm_open_v0(device)
        embedding_model = ESM3Wrapper(raw_backbone)
        tokenizer = EsmSequenceTokenizer()
        backbone_label = "esm3_sm_open_v1"
    else:
        model_name = f"facebook/{args.esm_model}"
        embedding_model = EsmModel.from_pretrained(model_name)
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        backbone_label = args.esm_model

    embedding_model.to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    inspect_checkpoint = checkpoint if not args.is_esm3 else {
        key: value for key, value in checkpoint.items()
        if not key.startswith("embedding_model.")
    }
    inferred_hidden, inferred_layers = infer_bilstm_params(inspect_checkpoint)
    hidden_size = inferred_hidden or args.hidden_size
    num_layers = inferred_layers or args.num_layers
    model = BiLSTMWithSelfAttentionModel(
        embedding_model=embedding_model,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=2,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model, tokenizer, backbone_label, hidden_size, num_layers


def check_payload(name: str, length: int, attention: np.ndarray, payload: dict) -> None:
    contribution = payload["matrix"]
    evidence = payload["intrinsic_signed_evidence"]
    influence = payload["signed_column_influence"]
    attention_mean = payload["attention_column_mean"]
    if attention.shape != (length, length) or contribution.shape != (length, length):
        raise ValueError(
            f"{name}: expected LxL matrices with L={length}; got "
            f"attention={attention.shape}, contribution={contribution.shape}"
        )
    for label, vector in (
        ("intrinsic_signed_evidence", evidence),
        ("signed_column_influence", influence),
        ("attention_column_mean", attention_mean),
        ("margin", payload["margin"]),
    ):
        if vector.shape != (length,):
            raise ValueError(f"{name}: {label} has shape {vector.shape}, expected {(length,)}")
    np.testing.assert_allclose(contribution, attention * evidence[None, :], rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(influence, contribution.mean(axis=0), rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(attention_mean, attention.mean(axis=0), rtol=2e-6, atol=2e-7)


def iter_json_write(handle, encoder: json.JSONEncoder, value) -> None:
    for chunk in encoder.iterencode(value):
        handle.write(chunk)


def extract_split(
    args: argparse.Namespace,
    split_name: str,
    split_csv: Path,
    output_path: Path,
    model,
    tokenizer,
    device: torch.device,
    backbone_label: str,
    hidden_size: int,
    num_layers: int,
) -> dict:
    frame = load_split(split_csv, args.max_proteins)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.part-{os.getpid()}")
    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    started = time.time()
    maximum_reconstruction_error = 0.0
    total_residues = 0

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "condition": args.condition,
        "seed": args.seed,
        "split": split_name,
        "source_csv": str(split_csv.resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "backbone": backbone_label,
        "architecture": "bilstm_attention",
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "class_definition": {
            "0": "rigid (Neq <= 1.0)",
            "1": "flexible (Neq > 1.0)",
        },
        "sign_definition": "positive supports flexible; negative supports rigid",
        "array_definitions": ARRAY_DEFINITIONS,
        "protein_count": int(len(frame)),
    }

    try:
        with gzip.open(
            temporary_path,
            mode="wt",
            encoding="utf-8",
            compresslevel=args.compression_level,
        ) as handle:
            handle.write("{")
            for index, (key, value) in enumerate(metadata.items()):
                if index:
                    handle.write(",")
                iter_json_write(handle, encoder, key)
                handle.write(":")
                iter_json_write(handle, encoder, value)
            handle.write(',"proteins":[')

            for protein_index, row in enumerate(frame.itertuples(index=False), start=1):
                name = row.name
                sequence = row.sequence
                attention, _tokens, _preds, _scores, _probs, payload = run_model_bilstm_attn(
                    model,
                    tokenizer,
                    sequence,
                    device,
                    task_type="classification",
                    return_logit_contributions=True,
                )
                if payload is None:
                    raise RuntimeError(f"{name}: exact-contribution payload was not returned")
                length = len(sequence)
                check_payload(name, length, attention, payload)
                maximum_reconstruction_error = max(
                    maximum_reconstruction_error,
                    float(payload["max_abs_error"]),
                )
                total_residues += length

                record = {
                    "name": name,
                    "sequence": sequence,
                    "length": length,
                    "contribution_matrix": payload["matrix"].tolist(),
                    "intrinsic_signed_evidence": payload["intrinsic_signed_evidence"].tolist(),
                    "signed_column_influence": payload["signed_column_influence"].tolist(),
                    "attention_matrix": attention.tolist(),
                    "attention_column_mean": payload["attention_column_mean"].tolist(),
                    "flex_minus_rigid_logit_margin": payload["margin"].tolist(),
                    "flex_minus_rigid_logit_margin_bias": payload["bias"],
                    "reconstruction_max_abs_error": payload["max_abs_error"],
                }
                if protein_index > 1:
                    handle.write(",")
                iter_json_write(handle, encoder, record)
                handle.flush()
                del record, payload, attention
                print(
                    f"[{args.condition} seed={args.seed} {split_name}] "
                    f"{protein_index}/{len(frame)} {name} L={length}",
                    flush=True,
                )
            handle.write("]}")
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    summary = {
        "split": split_name,
        "output": str(output_path.resolve()),
        "protein_count": int(len(frame)),
        "residue_count": int(total_residues),
        "max_reconstruction_abs_error": maximum_reconstruction_error,
        "elapsed_seconds": time.time() - started,
        "compressed_bytes": output_path.stat().st_size,
    }
    print("COMPLETED " + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this extraction")
    if torch.cuda.device_count() != 1:
        print(
            f"[warn] Expected one visible GPU per worker, found {torch.cuda.device_count()}; "
            "the current CUDA device will be used.",
            flush=True,
        )
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)
    print(f"Loading {args.condition} seed={args.seed} from {args.checkpoint}", flush=True)
    model, tokenizer, backbone, hidden_size, num_layers = load_model(args, device)

    output_dir = Path(args.output_dir)
    summary_path = output_dir / "extraction_summary.json"
    summaries_by_split = {}
    if args.resume and summary_path.is_file():
        previous = json.loads(summary_path.read_text())
        summaries_by_split = {
            item["split"]: item
            for item in previous.get("completed_splits", previous.get(
                "completed_splits_this_invocation", []
            ))
        }
    for split_name, csv_string in args.split:
        output_path = output_dir / f"{split_name}_signed_contributions.json.gz"
        if args.resume and output_path.is_file():
            print(f"SKIP existing atomic output: {output_path}", flush=True)
            continue
        summary = extract_split(
            args,
            split_name,
            Path(csv_string),
            output_path,
            model,
            tokenizer,
            device,
            backbone,
            hidden_size,
            num_layers,
        )
        summaries_by_split[split_name] = summary
    summary_path.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "condition": args.condition,
        "seed": args.seed,
        "completed_splits": [
            summaries_by_split[name]
            for name, _csv in args.split
            if name in summaries_by_split
        ],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
