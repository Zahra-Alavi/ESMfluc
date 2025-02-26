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

def main():
    parser = parse_arguments()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_training(args)

if __name__ == "__main__":
    main()
    
    

