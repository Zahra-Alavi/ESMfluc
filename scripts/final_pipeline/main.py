#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:43:57 2025

@author: zalavi
"""

import logging
from arguments import parse_arguments
from train import train

import torch 

def main():
    parser = parse_arguments()
    args = parser.parse_args()
    
    
    args.device = torch.device(args.device)


    logging.basicConfig(level=logging.INFO)
    train(args)

if __name__ == "__main__":
    main()
    
    

