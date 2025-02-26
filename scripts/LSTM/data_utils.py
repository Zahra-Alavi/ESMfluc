#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:00:07 2025

@author: zalavi
"""

import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import ast



# data_utils.py 

# =============================================================================
# Classification with Dynamic Thresholds 
# =============================================================================

def create_classification_func(num_classes, thresholds):
    """
    Given num_classes and a list of thresholds, return a function mapping
    a single Neq value to a class index.
    """
    sorted_thresholds = sorted(thresholds)

    def classify_neq(neq_value):
        for i, t in enumerate(sorted_thresholds):
            if neq_value <= t:
                return i
        return len(sorted_thresholds)

    # Sanity check:
    if len(thresholds) != (num_classes - 1):
        raise ValueError(
            f"Number of thresholds must be (num_classes - 1). Provided {len(thresholds)} "
            f"for {num_classes} classes."
        )
    return classify_neq


# =============================================================================
# Dataset and Collate
# =============================================================================

class SequenceClassificationDataset(Dataset):
    def __init__(self, encoded_inputs, labels):
        self.encoded_inputs = encoded_inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encoded_inputs[idx]['input_ids'].squeeze(0),
            'attention_mask': self.encoded_inputs[idx]['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx])
        }

def collate_fn_sequence(batch, tokenizer):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels_padded
    }


def load_and_preprocess_data(csv_path, classify_neq):
    """
    Loads CSV, drops invalid rows, applies classification function.
    Returns a pandas DataFrame with an extra column 'neq_class'.
    """
    # Load CSV
    training_data = pd.read_csv(csv_path)
    training_data['neq'] = training_data['neq'].apply(ast.literal_eval)

    # Drop invalid rows
    training_data = training_data.dropna(subset=['sequence', 'neq']).reset_index(drop=True)
    training_data = training_data[
        training_data['sequence'].str.len() == training_data['neq'].apply(len)
    ].reset_index(drop=True)

    # Classify
    training_data['neq_class'] = training_data['neq'].apply(
        lambda neq_list: [classify_neq(val) for val in neq_list]
    )
    
    return training_data