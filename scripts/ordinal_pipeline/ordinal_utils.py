import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


ORDINAL_THRESHOLDS = (1.0, 2.0, 3.0)
NUM_CLASSES = 4
NUM_THRESHOLDS = NUM_CLASSES - 1


def parse_neq_list(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return [float(v) for v in value]


def neq_to_class(v, thresholds=ORDINAL_THRESHOLDS):
    for i, t in enumerate(thresholds):
        if v <= t:
            return i
    return len(thresholds)


def neq_list_to_class_list(neq_values, thresholds=ORDINAL_THRESHOLDS):
    return [neq_to_class(v, thresholds) for v in neq_values]


def class_to_cumulative_binary(class_idx, num_classes=NUM_CLASSES):
    # 4 classes -> 3 binary thresholds
    # class 0 -> [0,0,0]
    # class 1 -> [1,0,0]
    # class 2 -> [1,1,0]
    # class 3 -> [1,1,1]
    return [1.0 if class_idx > k else 0.0 for k in range(num_classes - 1)]


def class_list_to_ordinal_targets(class_labels, num_classes=NUM_CLASSES):
    return [class_to_cumulative_binary(c, num_classes) for c in class_labels]


def load_ordinal_dataframe(csv_path, thresholds=ORDINAL_THRESHOLDS, max_len=1024):
    df = pd.read_csv(csv_path)
    df["neq"] = df["neq"].apply(parse_neq_list)

    df = df.dropna(subset=["sequence", "neq"]).reset_index(drop=True)
    df = df[df["sequence"].str.len() == df["neq"].apply(len)].reset_index(drop=True)

    if max_len is not None:
        long_mask = df["sequence"].str.len() > max_len
        n_long = int(long_mask.sum())
        if n_long:
            print(f"{n_long} sequences longer than {max_len} removed from {csv_path}")
            df = df[~long_mask].reset_index(drop=True)

    df["class_labels"] = df["neq"].apply(lambda x: neq_list_to_class_list(x, thresholds))
    df["ordinal_targets"] = df["class_labels"].apply(class_list_to_ordinal_targets)
    return df


class OrdinalSequenceDataset(Dataset):
    def __init__(self, encoded_inputs, class_labels, ordinal_targets):
        self.encoded_inputs = encoded_inputs
        self.class_labels = class_labels
        self.ordinal_targets = ordinal_targets

    def __len__(self):
        return len(self.class_labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encoded_inputs[idx]["input_ids"].squeeze(0),
            "attention_mask": self.encoded_inputs[idx]["attention_mask"].squeeze(0),
            "class_labels": torch.tensor(self.class_labels[idx], dtype=torch.long),
            "ordinal_targets": torch.tensor(self.ordinal_targets[idx], dtype=torch.float),
        }


def collate_fn_ordinal(batch, tokenizer):
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    class_labels = [item["class_labels"] for item in batch]
    ordinal_targets = [item["ordinal_targets"] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    class_labels_padded = pad_sequence(class_labels, batch_first=True, padding_value=-1)
    ordinal_targets_padded = pad_sequence(ordinal_targets, batch_first=True, padding_value=-1.0)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_masks_padded,
        "class_labels": class_labels_padded,
        "ordinal_targets": ordinal_targets_padded,
    }


def tokenize_sequences(sequences, tokenizer):
    return [
        tokenizer(seq, return_tensors="pt", padding=False, add_special_tokens=False)
        for seq in sequences
    ]


def enforce_monotonic_probs(probs):
    # Make P(y>1) >= P(y>2) >= P(y>3)
    mono_probs, _ = torch.cummin(probs, dim=-1)
    return mono_probs


def decode_ordinal_logits(logits, threshold=0.5):
    probs = torch.sigmoid(logits)
    probs = enforce_monotonic_probs(probs)
    # Count how many thresholds were passed -> final class 0..3
    preds = (probs > threshold).sum(dim=-1)
    return preds


def flatten_predictions(preds, targets, ignore_index=-1):
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    mask = targets != ignore_index
    return preds[mask].cpu().numpy(), targets[mask].cpu().numpy()


def compute_classification_outputs(all_true, all_pred):
    report = classification_report(all_true, all_pred, output_dict=True, digits=4)
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2, 3])
    return report, cm


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
