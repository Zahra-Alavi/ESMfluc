#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:43:57 2025

@author: zalavi
"""
# main.py 

import logging
from arguments import parse_arguments
from train2 import train

def dropout_rate_learning(args):
    print("----------------Running Dropout Rate Learning---------------")
    dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    for dropout in dropout_rates:
        args.dropout = dropout
        args.result_foldername = "dropout_{}".format(dropout)
        train(args)

def main():
    parser = parse_arguments()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.dropout_rate_learning:
        dropout_rate_learning(args)
    else:
        train(args)

if __name__ == "__main__":
    main()
    
    

