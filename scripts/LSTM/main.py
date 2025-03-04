#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:43:57 2025

@author: zalavi
"""
# main.py 

import logging
from arguments import parse_arguments
from train import run_training
from train2 import train
import itertools
import copy

def hyperparameter_search(args):
    param_grid = {
        "epochs": [10, 20, 30, 40, 50],
        "lr": [1e-5, 1e-4, 1e-3],
        "weight_decay": [1e-2, 1e-3, 1e-4],
        "esm_model": ["esm2_t6_8M_UR50D"],
        "dropout": [0.3, 0.5],
        "num_layers": range(1,10),
        "patience": [5, 10, 15],
        "hidden_size": [64, 128, 256, 512, 1024]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for idx, combo in enumerate(combinations):
        print(f"\nRunning hyperparameter search {idx + 1}/{len(combinations)} with settings:")
        print(combo)
        
        new_args = copy.deepcopy(args)
        for key, value in combo.items():
            setattr(new_args, key, value)

def main():
    parser = parse_arguments()
    args = parser.parse_args()
    
    if args.hyperparameter_search:
        hyperparameter_search(args)
        return

    logging.basicConfig(level=logging.INFO)
    # run_training(args)
    train(args)

if __name__ == "__main__":
    main()
    
    

