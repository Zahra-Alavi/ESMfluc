#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from transformers import EsmModel, EsmTokenizer

from ordinal_models import OrdinalBiLSTMModel
from ordinal_utils import (
    ORDINAL_THRESHOLDS,
    NUM_THRESHOLDS,
    OrdinalSequenceDataset,
    collate_fn_ordinal,
    compute_classification_outputs,
    decode_ordinal_logits,
    flatten_predictions,
    load_ordinal_dataframe,
    save_json,
    tokenize_sequences,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Train a self-contained ordinal regression model.")
    ap.add_argument("--train_data_file", type=str, default="../../data/train_data.csv")
    ap.add_argument("--test_data_file", type=str, default="../../data/test_data.csv")
    ap.add_argument("--esm_model", type=str, default="esm2_t12_35M_UR50D")
    ap.add_argument("--hidden_size", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--bidirectional", type=int, choices=[0, 1], default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--freeze_layers", type=str, default=None)
    ap.add_argument("--mixed_precision", action="store_true")
    ap.add_argument("--result_foldername", type=str, default="ordinal_bilstm")
    return ap.parse_args()


def freeze_esm_layers(embedding_model, freeze_layers):
    if not freeze_layers:
        return
    start_layer, end_layer = map(int, freeze_layers.split("-"))
    freeze_list = range(start_layer, end_layer + 1)

    for name, param in embedding_model.named_parameters():
        if "encoder.layer" in name:
            layer_num = int(name.split(".")[2])
            param.requires_grad = layer_num not in freeze_list
        else:
            param.requires_grad = True

    print(f"Freezing ESM layers {freeze_layers}")


def build_model(args):
    embedding_model = EsmModel.from_pretrained(f"facebook/{args.esm_model}")
    freeze_esm_layers(embedding_model, args.freeze_layers)

    model = OrdinalBiLSTMModel(
        embedding_model=embedding_model,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=bool(args.bidirectional),
        num_thresholds=NUM_THRESHOLDS,
    )
    return model.to(args.device)


def compute_pos_weight(ordinal_targets):
    flat = np.concatenate([np.array(x, dtype=np.float32) for x in ordinal_targets], axis=0)
    pos = flat.sum(axis=0)
    neg = flat.shape[0] - pos
    weights = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def masked_bce_with_logits(logits, targets, pos_weight):
    mask = targets != -1.0
    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))

    loss = F.binary_cross_entropy_with_logits(
        logits,
        safe_targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    loss = loss * mask.float()
    denom = mask.float().sum().clamp(min=1.0)
    return loss.sum() / denom


def evaluate(model, loader, device, pos_weight):
    model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            class_labels = batch["class_labels"].to(device)
            ordinal_targets = batch["ordinal_targets"].to(device)

            logits = model(input_ids, attention_mask)
            loss = masked_bce_with_logits(logits, ordinal_targets, pos_weight)
            total_loss += loss.item()

            preds = decode_ordinal_logits(logits)
            pred_flat, true_flat = flatten_predictions(preds, class_labels)
            all_pred.extend(pred_flat.tolist())
            all_true.extend(true_flat.tolist())

    avg_loss = total_loss / max(len(loader), 1)
    report, cm = compute_classification_outputs(all_true, all_pred)
    macro_f1 = report["macro avg"]["f1-score"]
    return avg_loss, report, cm


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
    run_dir = Path("./results_ordinal") / args.result_foldername
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = EsmTokenizer.from_pretrained(f"facebook/{args.esm_model}")

    train_df = load_ordinal_dataframe(args.train_data_file, thresholds=ORDINAL_THRESHOLDS, max_len=args.max_len)
    test_df = load_ordinal_dataframe(args.test_data_file, thresholds=ORDINAL_THRESHOLDS, max_len=args.max_len)

    train_df, val_df = train_test_split(
        train_df,
        test_size=args.val_fraction,
        random_state=42,
        shuffle=True,
    )

    X_train = tokenize_sequences(train_df["sequence"].tolist(), tokenizer)
    X_val = tokenize_sequences(val_df["sequence"].tolist(), tokenizer)
    X_test = tokenize_sequences(test_df["sequence"].tolist(), tokenizer)

    train_dataset = OrdinalSequenceDataset(
        X_train,
        train_df["class_labels"].tolist(),
        train_df["ordinal_targets"].tolist(),
    )
    val_dataset = OrdinalSequenceDataset(
        X_val,
        val_df["class_labels"].tolist(),
        val_df["ordinal_targets"].tolist(),
    )
    test_dataset = OrdinalSequenceDataset(
        X_test,
        test_df["class_labels"].tolist(),
        test_df["ordinal_targets"].tolist(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn_ordinal(x, tokenizer),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn_ordinal(x, tokenizer),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn_ordinal(x, tokenizer),
    )

    pos_weight = compute_pos_weight(train_df["ordinal_targets"].tolist()).to(args.device)

    model = build_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.mixed_precision and args.device.startswith("cuda")))

    best_val_macro_f1 = -1.0
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    val_macro_f1s = []

    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            ordinal_targets = batch["ordinal_targets"].to(args.device)

            optimizer.zero_grad()

            if args.mixed_precision and args.device.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask)
                    loss = masked_bce_with_logits(logits, ordinal_targets, pos_weight)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask)
                loss = masked_bce_with_logits(logits, ordinal_targets, pos_weight)
                loss.backward()
                optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        val_loss, val_report, _ = evaluate(model, val_loader, args.device, pos_weight)
        val_macro_f1 = val_report["macro avg"]["f1-score"]

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_macro_f1s.append(val_macro_f1)

        print(
            f"[Epoch {epoch}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), run_dir / "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=args.device))
    test_loss, test_report, test_cm = evaluate(model, test_loader, args.device, pos_weight)

    print(test_report)
    print(test_cm)

    pd.DataFrame(test_report).transpose().to_csv(run_dir / "classification_report.csv")
    save_json(test_report, run_dir / "classification_report.json")
    pd.DataFrame(test_cm, index=[0, 1, 2, 3], columns=[0, 1, 2, 3]).to_csv(
        run_dir / "confusion_matrix.csv"
    )
    save_confusion_matrix_png(test_cm, run_dir / "confusion_matrix.png")

    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Ordinal Training Curve")
    plt.legend()
    plt.savefig(run_dir / "loss_curve.png", bbox_inches="tight")
    plt.close()

    metrics = {
        "best_val_macro_f1": best_val_macro_f1,
        "test_loss": test_loss,
        "test_macro_f1": test_report["macro avg"]["f1-score"],
        "test_weighted_f1": test_report["weighted avg"]["f1-score"],
        "epochs_ran": len(train_losses),
        "total_seconds": time.perf_counter() - start_time,
    }
    save_json(metrics, run_dir / "metrics.json")

    config = vars(args).copy()
    config["device"] = str(args.device)
    save_json(config, run_dir / "config.json")


if __name__ == "__main__":
    main()
