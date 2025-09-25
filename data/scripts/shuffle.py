#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 10:41:37 2025

"""
import pandas as pd
import ast
import random


def shuffle_neq_in_row(neq_str):
    """
    Parse the 'neq' string into a list,
    shuffle it in place, then convert it back into a string.
    """
    # parse string -> list
    neq_list = ast.literal_eval(neq_str)
    random.shuffle(neq_list)
    # convert back to string
    return str(neq_list)

df = pd.read_csv("train_data.csv")
df["neq"] = df["neq"].apply(shuffle_neq_in_row)
df.to_csv("shuffled_inplace_data.csv", index=False)
