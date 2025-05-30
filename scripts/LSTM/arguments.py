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
    parser.add_argument("--train_data_file", type=str, default="../../data/train_data.csv",
                        help="Path to the training data CSV file.default=../../data/train_data.csv")

    parser.add_argument("--test_data_file", type=str, default="../../data/test_data.csv",
                        help="Path to the test data CSV file.default=../../data/test_data.csv")
    
    
    esm_models = [
        "esm1_t6_43M_UR50S",
        "esm1_t12_85M_UR50S",
        "esm1_t34_670M_UR100",
        "esm1_t34_670M_UR50D",
        "esm1_t34_670M_UR50S",
        "esm2_t6_8M_UR50D",
        "esm2_t12_35M_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t33_650M_UR50D",
        "esm2_t36_3B_UR50D",
        "esm2_t48_15B_UR50D"]
    
    parser.add_argument("--esm_model", type=str, default="esm2_t12_35M_UR50D",
    choices=esm_models,
    help=("Choose the ESM checkpoint to use."
          "Supported values include: " + ",".join(esm_models)+ ". Default is 'esm2_t12_35M_UR50D'."
    ),
)
    
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

# =============================================================================
#     # Oversampling
#     parser.add_argument("--oversampling", action="store_true",
#                         help="Enable oversampling of minority classes at sequence level. action='store_true")
# 
# =============================================================================
# =============================================================================
#     # Cross-validation
#     parser.add_argument("--cv_type", type=str, default="standard",
#                         choices=["stratified", "standard"],
#                         help="Use 'stratified' (StratifiedKFold) or 'standard' (KFold). default=stratified")
#     parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for cross-validation.default=5")
# =============================================================================
    

    # Layer freezing
    parser.add_argument("--freeze_layers", type=str, default=None,
                        help=("Specify which layers to freeze, e.g. '0-5' means freeze layers 0..5, "
                              "and unfreeze the rest. Omit or set None to not freeze any layers.default=None"))

    # Mixed precision
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Enable mixed precision original via torch.cuda.amp. action='store_true")
    
    # Learning Rate Scheduler
    parser.add_argument("--lr_scheduler", type=str, default="reduce_on_plateau",
                        choices=["none", "reduce_on_plateau"],
                        help="Use no scheduler ('none') or 'reduce_on_plateau' (PyTorch's ReduceLROnPlateau). default=reduce_on_plateau")
    parser.add_arguemtn("--dropout_rate_learning", 
                        action="store_true",
                        help="Getting the statistics for different dropout rates")
    parser.add_argument("--result_dir", type=str, default="../../results/",
                        help="Directory to save results. default=../../results/")

    return parser
