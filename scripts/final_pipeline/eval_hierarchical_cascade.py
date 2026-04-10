
import ast
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import EsmModel, EsmTokenizer

from models import (
    BiLSTMClassificationModel,
    BiLSTMWithSelfAttentionModel,
    TransformerClassificationModel,
    ESMLinearTokenClassifier,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True, type=str)
    ap.add_argument("--stage1_ckpt", required=True, type=str)
    ap.add_argument("--stage2_ckpt", required=True, type=str)
    ap.add_argument("--stage3_ckpt", required=True, type=str)
    ap.add_argument("--esm_model", default="esm2_t12_35M_UR50D", type=str)
    ap.add_argument("--architecture", default="bilstm", choices=["bilstm", "bilstm_attention", "transformer", "esm_linear"])
    ap.add_argument("--hidden_size", default=512, type=int)
    ap.add_argument("--num_layers", default=2, type=int)
    ap.add_argument("--dropout", default=0.3, type=float)
    ap.add_argument("--bidirectional", default=1, type=int)
    ap.add_argument("--head", default="softmax", type=str)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    ap.add_argument("--output_prefix", default="hierarchical_eval", type=str)
    return ap.parse_args()


def build_model(args):
    embedding_model = EsmModel.from_pretrained(f"facebook/{args.esm_model}")

    if args.architecture == "bilstm":
        model = BiLSTMClassificationModel(
            embedding_model=embedding_model,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=2,
            head=args.head,
            bidirectional=args.bidirectional,
        )
    elif args.architecture == "bilstm_attention":
        model = BiLSTMWithSelfAttentionModel(
            embedding_model=embedding_model,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=2,
            head=args.head,
            bidirectional=args.bidirectional,
        )
    elif args.architecture == "transformer":
        model = TransformerClassificationModel(
            embedding_model=embedding_model,
            num_classes=2,
            dropout=args.dropout,
            head=args.head,
        )
    elif args.architecture == "esm_linear":
        model = ESMLinearTokenClassifier(
            embedding_model=embedding_model,
            num_classes=2,
            head=args.head,
        )
    else:
        raise ValueError(args.architecture)

    return model


def load_checkpoint(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_binary(model, tokenizer, sequence, device):
    enc = tokenizer(sequence, return_tensors="pt", padding=False, add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask, return_features="pre")

    preds = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
    return preds.squeeze(0).cpu().tolist()


def neq_to_final4(neq_values):
    out = []
    for v in neq_values:
        if v <= 1.0:
            out.append(0)
        elif v <= 2.0:
            out.append(1)
        elif v <= 3.0:
            out.append(2)
        else:
            out.append(3)
    return out


def merge_cascade(stage1_preds, stage2_preds, stage3_preds):
    final = []
    for p1, p2, p3 in zip(stage1_preds, stage2_preds, stage3_preds):
        if p1 == 0:
            final.append(0)
        elif p2 == 0:
            final.append(1)
        elif p3 == 0:
            final.append(2)
        else:
            final.append(3)
    return final


def main():
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = EsmTokenizer.from_pretrained(f"facebook/{args.esm_model}")

    model1 = load_checkpoint(build_model(args), args.stage1_ckpt, device)
    model2 = load_checkpoint(build_model(args), args.stage2_ckpt, device)
    model3 = load_checkpoint(build_model(args), args.stage3_ckpt, device)

    df = pd.read_csv(args.test_csv)

    all_true = []
    all_pred = []
    per_sequence_rows = []

    for row in df.itertuples(index=False):
        seq = row.sequence
        neq = ast.literal_eval(row.neq)

        true4 = neq_to_final4(neq)
        pred1 = predict_binary(model1, tokenizer, seq, device)
        pred2 = predict_binary(model2, tokenizer, seq, device)
        pred3 = predict_binary(model3, tokenizer, seq, device)
        pred4 = merge_cascade(pred1, pred2, pred3)

        if not (len(seq) == len(true4) == len(pred4)):
            raise ValueError("Length mismatch in sequence/prediction/labels")

        all_true.extend(true4)
        all_pred.extend(pred4)

        per_sequence_rows.append({
            "sequence": seq,
            "true_4class": json.dumps(true4),
            "pred_4class": json.dumps(pred4),
        })

    report = classification_report(all_true, all_pred, output_dict=True, digits=4)
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2, 3])

    print(report)
    print(cm)

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    with open(f"{args.output_prefix}_classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    pd.DataFrame(cm, index=[0, 1, 2, 3], columns=[0, 1, 2, 3]).to_csv(
        f"{args.output_prefix}_confusion_matrix.csv"
    )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["=1", "(1,2]", "(2,3]", ">3"]
)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Hierarchical Cascade Confusion Matrix")
    plt.savefig(f"{args.output_prefix}_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    pd.DataFrame(per_sequence_rows).to_csv(
        f"{args.output_prefix}_per_sequence_predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
