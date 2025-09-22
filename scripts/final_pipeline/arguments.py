#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 09:58:50 2025

@author: zalavi
"""


import argparse

# =============================================================================
# Argument Parser
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a flexible ESM-based classification model.")

    # Data and basic setup
    parser.add_argument("--train_data_file", type=str, default="../data/train_data.csv",
                        help="Path to the training data CSV file.default=../data/train_data.csv")

    parser.add_argument("--test_data_file", type=str, default="../data/test_data.csv",
                        help="Path to the test data CSV file.default=../data/test_data.csv")

    # embedding model
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
          "Supported values include: " + ",".join(esm_models)+ ". Default is 'esm2_t12_35M_UR50D'."),
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
                        choices=["bilstm", "bilstm_attention", "transformer", "esm_linear"],
                        help="Model architecture: 'bilstm', 'bilstm_attention', or 'transformer'.")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size of LSTM layers. default=512")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers. default=2")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate. default=0.3")
    
    parser.add_argument("--transformer_nhead", type=int, default=8,
                        help="Number of attention heads in the Transformer encoder.")
    parser.add_argument("--transformer_num_encoder_layers", type=int, default=6,
                        help="Number of layers in the Transformer encoder.")
    parser.add_argument("--transformer_dim_feedforward", type=int, default=1024,
                        help="Dimension of the feedforward layer in the Transformer.")
    parser.add_argument("--bidirectional", type=int, choices=[0, 1], default=1,
                        help="Use bidirectional LSTM (1) or unidirectional (0). Default=1 (BiLSTM).")

    # Loss function
    parser.add_argument("--loss_function", type=str, default="focal",
                        choices=["focal", "crossentropy"],
                        help="Use 'focal' or 'crossentropy' loss. default=focal")
    # **Focal loss specific**: whether to compute class weights or not.
    parser.add_argument("--focal_class_weights", action="store_true",
                        help="If using focal loss, use class weights (alpha) computed from data.")
    
    # NC loss

    parser.add_argument("--head",
    choices=["centroid", "postfc", "softmax"], default="softmax",
    help="centroid = NC-only, no FC; postfc = NC on logits; "
         "softmax = NC on pre-FC + focal/CE")

    parser.add_argument("--lambda_nc1", type=float, default=1.0)
    parser.add_argument("--beta_nc2",   type=float, default=0.5)   # β
    parser.add_argument("--lambda_ce",  type=float, default=1.0)   # ignored in centroid

    parser.add_argument(
        "--loss_mode",
        type=str,
        choices=["supervised", "nc", "both"],  # supervised = CE/Focal only; nc = NC only; both = NC + CE/Focal
        default="both",
        help="Select loss composition: 'supervised' (CE/Focal only), 'nc' (NC losses only), or 'both' (NC + CE/Focal). default=both")
    
    

    # Oversampling
    parser.add_argument("--oversampling", action="store_true",
                        help="Enable oversampling of minority classes at sequence level. action='store_true")
    
    parser.add_argument("--oversampling_threshold", type=float, default=0.1, help="Fraction of minority residues required for oversampling. default=0.1")
    
    parser.add_argument("--undersampling_threshold", type=float, default=0.01, help="Fraction of minority residues below which to down-weight samples. default=0.1")
    
    parser.add_argument("--undersampling_intensity", type=float, default=0.1, help="scaling factor: how much to undersample majority-class sequences. default=0.1")
    
    parser.add_argument("--oversampling_intensity", type=float, default=5.0, help="scaling factor: how much to oversample minority-class sequences. default=5.0")
    
    
 
    # Whether to wrap the model in DataParallel if multiple GPUs are available
    parser.add_argument("--data_parallel", action="store_true",
                       help="Use nn.DataParallel for multi-GPU training if more than one GPU is available.")


    
    
    parser.add_argument("--warmup_epochs", type=int, default=0,
    help="Number of epochs to keep lambda_ce > 0 before setting it to 0 (centroid/postfc only)")
    
    
    


    # Layer freezing
    parser.add_argument("--freeze_all_backbone", action="store_true",
                        help="Freeze ALL parameters in the ESM backbone. Overrides --freeze_layers.") 
    
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
    
    parser.add_argument('--device', default="cuda",
                    help="Which device string to pass to torch.device")

    parser.add_argument("--amp_dtype", type=str, choices=["fp16", "bf16"], default="fp16",
                    help="Autocast dtype when using mixed precision on CUDA. Default=fp16")

    
    
    # reproducibility
    parser.add_argument("--seed", type=int, default=42,
                    help="Global random seed for full reproducibility. default=42")

    parser.add_argument("--result_foldername", type=str, default="timestamp",
                        help="Name for the result folder. default=timestamp")
# =============================================================================
#     parser.add_argument("--deterministic", action="store_true",
#                     help="Use fully deterministic CUDA/cuDNN kernels (slower, may raise on some ops).")
# =============================================================================


    return parser