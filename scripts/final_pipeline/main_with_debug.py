#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:43:57 2025

"""

import logging
import sys
import traceback
from arguments import parse_arguments
from train import train, train_regression

import torch 

def main():
    try:
        print("=== MAIN START ===", flush=True)
        parser = parse_arguments()
        print("Parser created", flush=True)
        
        args = parser.parse_args()
        print(f"Args parsed: task_type={args.task_type}", flush=True)
        
        args.device = torch.device(args.device)
        print(f"Device set: {args.device}", flush=True)

        logging.basicConfig(level=logging.INFO)
        
        print(f"Starting training with task_type={args.task_type}", flush=True)
        
        # Route to appropriate training function based on task type
        if args.task_type == "classification":
            print("Calling train()", flush=True)
            train(args)  # Existing classification training
        elif args.task_type == "regression":
            print("Calling train_regression()", flush=True)
            train_regression(args)  # New regression training
        else:
            raise ValueError(f"Unknown task_type: {args.task_type}")
            
        print("=== TRAINING COMPLETED ===", flush=True)
        
    except Exception as e:
        print(f"\n=== ERROR ===", file=sys.stderr, flush=True)
        print(f"Exception: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
