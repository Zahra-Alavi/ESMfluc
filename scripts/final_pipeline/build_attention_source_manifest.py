#!/usr/bin/env python3
"""
Build an expanded manifest where each row is one attention source.

Input manifest rows are model runs.  The expanded manifest separates:
  - ESM2 frozen linear transformer attention
  - ESM2 BiLSTM attention for frozen/top4/top28
  - ESM2 backbone transformer attention for frozen/top4/top28
  - ESM3 BiLSTM attention for frozen/top4/top28

This lets the existing analysis scripts treat each attention source as a
condition without changing their manifest parser.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


CONDITION_MAP = {
    "esm2_frozen_linear": "esm2_frozen_linear_transformer",
    "esm2_frozen_bilstm_attn": "esm2_frozen_bilstm_attention",
    "esm2_top4_bilstm_attn": "esm2_top4_bilstm_attention",
    "esm2_top28_bilstm_attn": "esm2_top28_bilstm_attention",
    "esm3_frozen_bilstm_attn": "esm3_frozen_bilstm_attention",
    "esm3_top4_bilstm_attn": "esm3_top4_bilstm_attention",
    "esm3_top28_bilstm_attn": "esm3_top28_bilstm_attention",
}

ESM2_BACKBONE_CONDITIONS = {
    "esm2_frozen_bilstm_attn": "esm2_frozen_backbone_attention",
    "esm2_top4_bilstm_attn": "esm2_top4_backbone_attention",
    "esm2_top28_bilstm_attn": "esm2_top28_backbone_attention",
}


def parse_args():
    p = argparse.ArgumentParser(description="Build 10-condition attention-source manifest.")
    p.add_argument("--source_result_root", required=True)
    p.add_argument("--output_result_root", required=True)
    p.add_argument("--backbone_filename", default="backbone_attention.json")
    p.add_argument(
        "--write_source_copy",
        action="store_true",
        help="Also write source_result_root/manifest_attention_sources.tsv.",
    )
    return p.parse_args()


def copy_reference_inputs(source_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in [
        "test_data.csv",
        "test_data_with_names.csv",
        "test_data_sequences.fasta",
        "contact_maps_ca8.json",
    ]:
        src = source_root / name
        if src.exists():
            shutil.copy2(src, output_root / name)


def require_one(df: pd.DataFrame, condition: str, seed: int) -> pd.Series:
    sub = df[(df["condition"] == condition) & (df["seed"].astype(int) == int(seed))]
    if len(sub) != 1:
        raise ValueError(f"Expected one row for {condition} seed {seed}, found {len(sub)}")
    return sub.iloc[0]


def make_row(source_row: pd.Series, condition: str, attention_json: Path, attention_kind: str) -> dict:
    row = source_row.to_dict()
    row["condition"] = condition
    row["attention_json"] = str(attention_json.resolve())
    row["attention_kind"] = attention_kind
    row["source_condition"] = source_row["condition"]
    row["source_attention_json"] = source_row["attention_json"]
    return row


def backbone_attention_path(source_row: pd.Series, filename: str) -> Path:
    value = source_row.get("backbone_attention_json", "")
    if pd.notna(value) and str(value).strip():
        return Path(str(value))
    return Path(source_row["run_dir"]) / filename


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_result_root).expanduser().resolve()
    output_root = Path(args.output_result_root).expanduser().resolve()
    manifest_path = source_root / "manifest.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path, sep="\t")
    seeds = sorted(manifest["seed"].astype(int).unique())
    rows = []

    for seed in seeds:
        for source_condition, expanded_condition in CONDITION_MAP.items():
            src = require_one(manifest, source_condition, seed)
            rows.append(
                make_row(
                    src,
                    expanded_condition,
                    Path(src["attention_json"]),
                    "esm2_transformer_linear_attention"
                    if source_condition == "esm2_frozen_linear"
                    else "bilstm_attention",
                )
            )

        for source_condition, expanded_condition in ESM2_BACKBONE_CONDITIONS.items():
            src = require_one(manifest, source_condition, seed)
            backbone_path = backbone_attention_path(src, args.backbone_filename)
            rows.append(
                make_row(
                    src,
                    expanded_condition,
                    backbone_path,
                    "esm2_backbone_last_layer_avg_heads",
                )
            )

    expanded = pd.DataFrame(rows)
    expanded = expanded.sort_values(["condition", "seed"]).reset_index(drop=True)

    copy_reference_inputs(source_root, output_root)
    expanded.to_csv(output_root / "manifest.tsv", sep="\t", index=False)
    if args.write_source_copy:
        expanded.to_csv(source_root / "manifest_attention_sources.tsv", sep="\t", index=False)

    missing = [p for p in expanded["attention_json"].map(Path) if not p.exists()]
    print(f"Source root: {source_root}")
    print(f"Output root: {output_root}")
    print(f"Rows: {len(expanded)}")
    print(f"Conditions: {expanded['condition'].nunique()}")
    print(f"Seeds: {', '.join(map(str, seeds))}")
    print(f"Missing attention JSONs: {len(missing)}")
    for path in missing[:20]:
        print(f"  missing: {path}")


if __name__ == "__main__":
    main()
