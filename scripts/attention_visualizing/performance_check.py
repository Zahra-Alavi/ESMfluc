#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 18:16:46 2025

@author: zalavi
"""
import pandas as pd

"""
This script takes test_data_with_predictions.json and check the model performance. 
note: test_data_with_predictions.json is the output of predict.py
"""
test_preds = pd.read_json("../../results/test_data_with_predictions.json", orient="records")

def check_lengths(row):
    return len(row['sequence']) == len(row['neq']) == len(row['neq_class']) ==len(row['pred_class'])

# check everything has the same length
length_ok = test_preds.apply(check_lengths, axis=1)
invalid_rows = test_preds[~length_ok]
print(f"Number of rows with length mismatch: {len(invalid_rows)}")  # should be zero


def get_mismatch_details(row):
    seq = row['sequence']
    neq_values = row['neq']           
    gt_classes = row['neq_class']      
    preds = row['pred_class']          

    mismatch ={}
    for i in range(len(seq)):
        if gt_classes[i] != preds[i]:
            mismatch[i + 1] = (seq[i], neq_values[i])
    return mismatch

test_preds['mismatch_details'] = test_preds.apply(get_mismatch_details, axis=1)



all_gt = []
all_preds = []
for _, row in test_preds.iterrows():
    all_gt.extend(row['neq_class'])
    all_preds.extend(row['pred_class'])


correct = sum(int(g == p) for g, p in zip(all_gt, all_preds))
total = len(all_gt)
accuracy = correct / total

print(f"Overall accuracy: {accuracy:.3f} ({correct}/{total})")


from sklearn.metrics import classification_report
print(classification_report(all_gt, all_preds))


import collections

mistakes_per_class = collections.defaultdict(list)

for _, row in test_preds.iterrows():
    seq = row['sequence']
    eq_list = row['neq']
    gt = row['neq_class']
    preds = row['pred_class']
    for i in range(len(seq)):
        if gt[i] != preds[i]:
            # ground truth class
            c = gt[i]
            mistakes_per_class[c].append(eq_list[i])

# plot histograms per class:
import matplotlib.pyplot as plt

for c, values in mistakes_per_class.items():
    plt.figure()
    plt.hist(values, bins=30)
    plt.title(f"Misclassified NEQ distribution for GT class = {c}")
    plt.xlabel("NEQ value")
    plt.ylabel("Count of misclassifications")
    plt.show()

