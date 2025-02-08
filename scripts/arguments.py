#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:58:50 2025

@author: zalavi
"""
# arguments.py 

import argparse

# =============================================================================
# Argument Parser
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a flexible ESM-based classification model.")

    # Data and basic setup
    parser.add_argument("--csv_path", type=str, default="data/neq_training_data.csv",
                        help="Path to the input CSV containing sequences and Neq values.default=neq_training_data.csv")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size. default=4")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs. default=20")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience. default=5")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for AdamW. default=1e-5")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW. default=1e-2")

    # Classification thresholds
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Total number of classes for Neq classification.default=4")
    parser.add_argument("--neq_thresholds", nargs="*", type=float, default=[1.0, 2.0, 4.0],
                        help=("Thresholds for dividing Neq values. "
                              "If num_classes=4, you might use 1.0 2.0 4.0. "
                              "For num_classes=3, maybe 1.0 1.5, etc." "default=[1.0, 2.0, 4.0]"))

    # Architecture
    parser.add_argument("--architecture", type=str, default="bilstm",
                        choices=["bilstm", "bilstm_attention"],
                        help="Choose between BiLSTM ('bilstm') or BiLSTM+SelfAttention ('bilstm_attention'). default='bilstm")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size of LSTM layers. default=512")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers. default=2")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate. default=0.3")

    # Loss function
    parser.add_argument("--loss_function", type=str, default="focal",
                        choices=["focal", "crossentropy"],
                        help="Use 'focal' or 'crossentropy' loss. default=focal")
    # **Focal loss specific**: whether to compute class weights or not.
    parser.add_argument("--focal_class_weights", action="store_true",
                        help="If using focal loss, use class weights (alpha) computed from data.")

    # Oversampling
    parser.add_argument("--oversampling", action="store_true",
                        help="Enable oversampling of minority classes at sequence level. action='store_true")

    # Cross-validation
    parser.add_argument("--cv_type", type=str, default="stratified",
                        choices=["stratified", "standard"],
                        help="Use 'stratified' (StratifiedKFold) or 'standard' (KFold). default=stratified")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for cross-validation.default=5")

    # Layer freezing
    parser.add_argument("--freeze_layers", type=str, default=None,
                        help=("Specify which layers to freeze, e.g. '0-5' means freeze layers 0..5, "
                              "and unfreeze the rest. Omit or set None to not freeze any layers.default=None"))

    # Mixed precision
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Enable mixed precision training via torch.cuda.amp. action='store_true")
    
    # Learning Rate Scheduler
    parser.add_argument("--lr_scheduler", type=str, default="reduce_on_plateau",
                        choices=["none", "reduce_on_plateau"],
                        help="Use no scheduler ('none') or 'reduce_on_plateau' (PyTorch's ReduceLROnPlateau). default=reduce_on_plateau")

    return parser