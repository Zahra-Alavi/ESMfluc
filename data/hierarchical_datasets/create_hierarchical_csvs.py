import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


def create_hierarchical_stage_csvs(
    input_csv,
    output_dir,
    sequence_col="sequence",
    neq_col="neq",
    keep_cols=None,
    drop_all_masked=True,
):
    """
    Create 3 stage-specific CSVs for a true hierarchical cascade.

    Stage 1:
        0 if neq <= 1
        1 if neq > 1

    Stage 2:
        -1 if neq <= 1   (masked)
         0 if 1 < neq <= 2
         1 if neq > 2

    Stage 3:
        -1 if neq <= 2   (masked)
         0 if 2 < neq <= 3
         1 if neq > 3

    Also writes a final 4-class ground-truth file:
        0 if neq <= 1
        1 if 1 < neq <= 2
        2 if 2 < neq <= 3
        3 if neq > 3

    Output CSVs contain:
        sequence
        labels   <- list of ints as a string, parseable with ast.literal_eval

    Notes:
    - `-1` means "ignore this residue for this stage".
    - If `drop_all_masked=True`, sequences with no active residues for a stage are removed.
    """

    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if keep_cols is None:
        keep_cols = []

    df = pd.read_csv(input_csv)

    def parse_neq_list(x):
        if isinstance(x, list):
            arr = np.array(x, dtype=float)
        elif pd.isna(x):
            return None
        else:
            # Handles strings containing bare 'nan'
            arr = np.array(eval(str(x), {"__builtins__": {}}, {"nan": float("nan")}), dtype=float)

        if np.isnan(arr).any():
            return None
        return arr.tolist()

    def stage1_labels(vals):
        return [0 if v <= 1.0 else 1 for v in vals]

    def stage2_labels(vals):
        out = []
        for v in vals:
            if v <= 1.0:
                out.append(-1)
            elif v <= 2.0:
                out.append(0)
            else:
                out.append(1)
        return out

    def stage3_labels(vals):
        out = []
        for v in vals:
            if v <= 2.0:
                out.append(-1)
            elif v <= 3.0:
                out.append(0)
            else:
                out.append(1)
        return out

    def final4_labels(vals):
        out = []
        for v in vals:
            if v <= 1.0:
                out.append(0)
            elif v <= 2.0:
                out.append(1)
            elif v <= 3.0:
                out.append(2)
            else:
                out.append(3)
        return out

    df = df.copy()
    df["neq_values"] = df[neq_col].apply(parse_neq_list)
    df = df.dropna(subset=[sequence_col, "neq_values"]).reset_index(drop=True)
    df = df[df[sequence_col].str.len() == df["neq_values"].apply(len)].reset_index(drop=True)

    base_cols = [c for c in keep_cols if c in df.columns] + [sequence_col]

    stage_builders = {
        "stage1": stage1_labels,
        "stage2": stage2_labels,
        "stage3": stage3_labels,
        "final4": final4_labels,
    }

    written = {}

    for name, fn in stage_builders.items():
        out_df = df[base_cols].copy()
        out_df["labels"] = df["neq_values"].apply(fn)

        if drop_all_masked and name in {"stage2", "stage3"}:
            out_df = out_df[out_df["labels"].apply(lambda x: any(v != -1 for v in x))].reset_index(drop=True)

        out_df["labels"] = out_df["labels"].apply(json.dumps)

        out_path = output_dir / f"{input_csv.stem}_{name}.csv"
        out_df.to_csv(out_path, index=False)
        written[name] = out_path

        print(f"{name}: wrote {out_path} ({len(out_df)} rows)")

    return written

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    create_hierarchical_stage_csvs(
        input_csv=HERE / "../../data/train_data.csv",
        output_dir=HERE,
        sequence_col="sequence",
        neq_col="neq",
    )

    create_hierarchical_stage_csvs(
        input_csv=HERE / "../../data/test_data.csv",
        output_dir=HERE,
        sequence_col="sequence",
        neq_col="neq",
    )

