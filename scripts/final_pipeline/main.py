#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:43:57 2025

"""

import logging
from arguments import parse_arguments
import torch 

def main():
    parser = parse_arguments()
    args = parser.parse_args()
    
    args.device = torch.device(args.device)

    logging.basicConfig(level=logging.INFO)

    # Route ordinal task directly to train.py implementation
    if args.task_type == "ordinal":
        from train import train_ordinal
        print("Using train.py (ordinal)")
        train_ordinal(args)
        return
    
    # Import from train_unified if available, otherwise fall back to train
    try:
        from train_unified import train_model
        print("Using train_unified.py for training")
        train_model(args)
    except ImportError:
        print("train_unified.py not found, using train.py")
        from train import train, train_regression
        
        # Route to appropriate training function based on task type
        if args.task_type == "classification":
            train(args)  # Existing classification training
        elif args.task_type == "regression":
            train_regression(args)  # New regression training
        else:
            raise ValueError(f"Unknown task_type: {args.task_type}")

if __name__ == "__main__":
    main()
