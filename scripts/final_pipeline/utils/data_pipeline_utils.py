#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from .backbone_utils import build_tokenizer, tokenize
from .data_utils import (
    SequenceClassificationDataset,
    collate_fn_sequence,
    compute_sampling_weights,
    create_classification_func,
    load_and_preprocess_data,
)


def prepare_data_for_training(args, drop_last: bool):
    labeled_neq = create_classification_func(args.num_classes, args.neq_thresholds)
    train_data = load_and_preprocess_data(args.train_data_file, labeled_neq)
    test_data = load_and_preprocess_data(args.test_data_file, labeled_neq)

    tokenizer = build_tokenizer(args.esm_model)
    X_train = tokenize(train_data["sequence"], tokenizer)
    X_test = tokenize(test_data["sequence"], tokenizer)
    y_train = train_data["neq_class"].tolist()
    y_test = test_data["neq_class"].tolist()

    val_loader = None
    need_val = (args.lr_scheduler == "reduce_on_plateau") or (args.patience and args.patience > 0)

    if need_val:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            stratify=None,
            random_state=args.seed,
        )
        val_dataset = SequenceClassificationDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=lambda x: collate_fn_sequence(x, tokenizer),
        )

    train_dataset = SequenceClassificationDataset(X_train, y_train)
    test_dataset = SequenceClassificationDataset(X_test, y_test)

    raw_labels = []
    for i in range(len(train_dataset)):
        raw_labels.extend(train_dataset.labels[i])
    print("Original Class Distribution:", Counter(raw_labels))

    flat = [lab for seq in train_dataset.labels for lab in seq]
    occurrence_list = [flat.count(k) for k in range(args.num_classes)]
    weight_factor = torch.tensor(
        [1.0 / np.sqrt(max(n, 1)) for n in occurrence_list],
        dtype=torch.float,
        device=args.device,
    )

    if args.oversampling:
        print("Applying oversampling using WeightedRandomSampler...")
        sampling_weights = compute_sampling_weights(
            train_dataset,
            num_classes=args.num_classes,
            neq_thresholds=args.neq_thresholds,
            oversampling_threshold=args.oversampling_threshold,
            undersampling_threshold=args.undersampling_threshold,
            undersampling_intensity=args.undersampling_intensity,
            oversampling_intensity=args.oversampling_intensity,
        )

        sampler = WeightedRandomSampler(
            weights=sampling_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=lambda x: collate_fn_sequence(x, tokenizer),
            drop_last=drop_last,
        )

        oversampled_labels = []
        for batch in train_loader:
            batch_labels = batch["labels"].cpu().numpy().flatten()
            batch_labels = batch_labels[batch_labels != -1]
            oversampled_labels.extend(batch_labels)
        print("Sampled Class Distribution After Oversampling:", Counter(oversampled_labels))
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn_sequence(x, tokenizer),
            drop_last=drop_last,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn_sequence(x, tokenizer),
        drop_last=False,
    )

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "occurrence_list": occurrence_list,
        "weight_factor": weight_factor,
    }
