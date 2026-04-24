#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from transformers import EsmModel, EsmTokenizer

from ordinal_models import OrdinalBiLSTMModel
from ordinal_utils import (
    NUM_THRESHOLDS,
    OrdinalSequenceDataset,
    collate_fn_ordinal,
    compute_classification_outputs,
    decode_ordinal_logits,
    flatten_predictions,
    load_ordinal_dataframe,
    tokenize_sequences,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate a trained ordinal regression model.")
    ap.add_argument("--checkpoint_dir", required=True, type=str)
    ap.add_argument("--test_data_file", default=None, type=str)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    return ap.parse_args()


def build_model(config, device):
    embedding_model = EsmModel.from_pretrained(f"facebook/{config['esm_model']}")
    model = OrdinalBiLSTMModel(
        embedding_model=embedding_model,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        bidirectional=bool(config["bidirectional"]),
        num_thresholds=NUM_THRESHOLDS,
    )
    return model.to(device)


def save_confusion_matrix_png(cm, output_path):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["<=1", "(1,2]", "(2,3]", ">3"],
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Ordinal Regression Confusion Matrix")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)

    with open(ckpt_dir / "config.json") as f:
        config = json.load(f)

    test_data_file = args.test_data_file or config["test_data_file"]
    device = args.device

    tokenizer = EsmTokenizer.from_pretrained(f"facebook/{config['esm_model']}")
    test_df = load_ordinal_dataframe(test_data_file, max_len=config["max_len"])

    X_test = tokenize_sequences(test_df["sequence"].tolist(), tokenizer)
    test_dataset = OrdinalSequenceDataset(
        X_test,
        test_df["class_labels"].tolist(),
        test_df["ordinal_targets"].tolist(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=lambda x: collate_fn_ordinal(x, tokenizer),
    )

    model = build_model(config, device)
    model.load_state_dict(torch.load(ckpt_dir / "best_model.pth", map_location=device))
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            class_labels = batch["class_labels"].to(device)

            logits = model(input_ids, attention_mask)
            preds = decode_ordinal_logits(logits)

            pred_flat, true_flat = flatten_predictions(preds, class_labels)
            all_pred.extend(pred_flat.tolist())
            all_true.extend(true_flat.tolist())

    report, cm = compute_classification_outputs(all_true, all_pred)

    print(report)
    print(cm)

    pd.DataFrame(report).transpose().to_csv(ckpt_dir / "classification_report_eval.csv")
    with open(ckpt_dir / "classification_report_eval.json", "w") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(cm, index=[0, 1, 2, 3], columns=[0, 1, 2, 3]).to_csv(
        ckpt_dir / "confusion_matrix_eval.csv"
    )
    save_confusion_matrix_png(cm, ckpt_dir / "confusion_matrix_eval.png")


if __name__ == "__main__":
    main()
