#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:43:57 2025

"""

import logging
from arguments import parse_arguments
from train import train, train_regression

import torch 

def main():
    parser = parse_arguments()
    args = parser.parse_args()
    
    args.device = torch.device(args.device)

    logging.basicConfig(level=logging.INFO)
    
    # Route to appropriate training function based on task type
    if args.task_type == "classification":
        train(args)  # Existing classification training
    elif args.task_type == "regression":
        train_regression(args)  # New regression training
    else:
        raise ValueError(f"Unknown task_type: {args.task_type}")
