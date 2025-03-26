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
import numpy as np


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
    def __init__(self, encoded_inputs, labels, sequences, neq_values):
        self.sequences = sequences
        self.encoded_inputs = encoded_inputs
        self.neq_values = neq_values
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'neq': self.neq_values[idx],
            'input_ids': self.encoded_inputs[idx]['input_ids'].squeeze(0),
            'attention_mask': self.encoded_inputs[idx]['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx])
        }

def collate_fn_sequence(batch, tokenizer):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    sequences = [item['sequence'] for item in batch]
    neq_values = [item['neq'] for item in batch]

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value='')
    neq_values_padded = pad_sequence(neq_values, batch_first=True, padding_value=-1)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'labels': labels_padded,
        'sequences': sequences_padded,
        'neq_values': neq_values_padded
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

# =============================================================================
# Compute Sampling Weights
# =============================================================================

def compute_sampling_weights(dataset, num_classes, neq_thresholds, oversampling_threshold=0.10, undersampling_threshold=0.05, oversampling_intensity = 5.0, undersampling_intensity = 0.01):
    """
    Compute sampling weights for each sequence based on the fraction of minority residues.
    - Dynamically identifies minority classes based on neq_thresholds.
    - Oversamples sequences rich in minority classes.
    - Undersamples sequences dominated by majority classes.

    Arguments:
        dataset: PyTorch dataset containing sequences and labels.
        num_classes (int): Total number of classes.
        neq_thresholds (list): List of thresholds used for classification.
        oversampling_threshold (float): Fraction of minority residues required for oversampling.
        undersampling_threshold (float): Fraction of minority residues below which to down-weight samples.
        undersampling_intensity (float): Scaling factor: how much to undersample majority-class sequences.
        oversampling_intensity (float): Scaling factor: how much to oversample minority-class sequences.

    Returns:
        Tensor of sampling weights.
    """
    # Identify minority and majority classes based on threshold distribution
    class_counts = np.zeros(num_classes)

    # Compute class counts across dataset
    for labels in dataset.labels:
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            class_counts[u] += c

    # Define minority classes as those with fewer occurrences than the median class
    median_count = np.median(class_counts)
    minority_classes = [i for i in range(num_classes) if class_counts[i] < median_count]
    
    weights = []
    
    for i in range(len(dataset)):
        labels = dataset.labels[i]
        total_residues = len(labels)

        # Compute fraction of residues belonging to minority classes
        minority_fraction = sum(1 for x in labels if x in minority_classes) / total_residues

        # Assign weights dynamically
        if minority_fraction == 0:
            weight = 0.001  # ignore sequences that only have class 0 & 1
        if minority_fraction < undersampling_threshold:
            weight = undersampling_intensity  # Undersample sequences with almost no minority residues
        else:
           #weight = 1.0 + oversampling_intensity * (minority_fraction / oversampling_threshold) # Oversample minority-rich sequences
           weight = 1

        weights.append(weight)

    # Normalize weights to sum to dataset size
    weights = np.array(weights)
    weights = weights / np.sum(weights) * len(weights)  # Normalize
    return torch.tensor(weights, dtype=torch.float)