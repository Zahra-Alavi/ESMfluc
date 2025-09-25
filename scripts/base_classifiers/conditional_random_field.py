"""
Description: This only one file used for implementing a Conditional Random Field (CRF) model for predicting Neq values of amino acids in protein sequences.
Date: 2025-02-08
"""

import os
import ast
import itertools
import argparse
import pandas as pd
import numpy as np
import sklearn_crfsuite
import sklearn_crfsuite.metrics
import time
from decimal import Decimal
from tabulate import tabulate

def extract_features(sequence, window_size, characteristics_dict, selected_features):
    features = []
    seq_length = len(sequence)

    for i in range(seq_length):
        feature_dict = {
            "residue": sequence[i],
            "prev_residue": sequence[i - 1] if i > 0 else "START",
            "next_residue": sequence[i + 1] if i < seq_length - 1 else "END",
        }
        for feature in selected_features:
            value = characteristics_dict.get(sequence[i], {}).get(feature.lower(), None)
            if value is None:
                print(f"Warning: Feature '{feature}' for residue '{sequence[i]}' is missing.")
            feature_dict[feature] = value

        for w in range(1, window_size + 1):
            feature_dict[f"prev_{w}_residue"] = sequence[i - w] if i - w >= 0 else "START"
            feature_dict[f"next_{w}_residue"] = sequence[i + w] if i + w < seq_length else "END"

        features.append(feature_dict)
    
    return features

def classify_neq(neq_values):
    return ["0" if Decimal(neq) == Decimal("1.0") else "1" for neq in neq_values]

def load_data(amino_acids_file, train_data_file):
    amino_acids_characteristics = pd.read_csv(amino_acids_file)
    characteristics_dict = {
        row['amino_acids']: row.drop('amino_acids').to_dict()
        for _, row in amino_acids_characteristics.iterrows()
    }

    data = pd.read_csv(train_data_file)
    sequences = data["sequence"].tolist()
    labels = data['neq'].apply(lambda x: list(map(float, ast.literal_eval(x)))).apply(classify_neq)
    return sequences, labels.tolist(), characteristics_dict

def tune_hyperparameters(X_train, y_train, X_test, y_test, show_progress):
    best_accuracy, best_c1, best_c2 = 0, 0, 0
    c1_values, c2_values = [0.001, 0.01, 0.1, 1, 10, 100], [0.001, 0.01, 0.1, 1, 10, 100]
    results = []

    for c1, c2 in itertools.product(c1_values, c2_values):
        crf = train_crf(X_train, y_train, c1, c2)
        accuracy = evaluate_crf(crf, X_test, y_test)
        results.append([c1, c2, accuracy])

        if accuracy > best_accuracy:
            best_accuracy, best_c1, best_c2 = accuracy, c1, c2
        
        if show_progress:
            print(f"Trying c1={c1}, c2={c2} -> Accuracy: {accuracy:.4f}")
    if show_progress:
        print("\nHyperparameter Tuning Results:")
        print(tabulate(results, headers=["c1", "c2", "Accuracy"], tablefmt="grid"))
    return best_c1, best_c2

def train_crf(X_train, y_train, c1=0.1, c2=0.1):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=c1,
        c2=c2,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    print("Max iterations allowed:", crf.max_iterations)
    return crf

def evaluate_crf(crf, X_test, y_test):
    y_pred = crf.predict(X_test)
    return sklearn_crfsuite.metrics.flat_accuracy_score(y_test, y_pred)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a CRF model for predicting Neq values of amino acids.")
    parser.add_argument('--amino_acids_file', type=str, default="../../data/amino_acids_characteristics.csv", help='Path to the amino acids characteristics CSV file.')
    parser.add_argument('--train_data_file', type=str, default="../../data/train_data.csv", help='Path to the training data CSV file.')
    parser.add_argument('--test_data_file', type=str, default="../../data/test_data.csv", help='Path to the test data CSV file.')
    parser.add_argument('--window_size', type=int, default=2, help='Window size for feature extraction (default is 2).')
    parser.add_argument('--features', type=str, default="charges,polar,hydrophobic", help='Comma-separated list of features to use (choices: charges, polar, hydrophobic, molecular_weight, pKa, pKb, pKx, pI).')
    parser.add_argument('--hyperparameter_tuning', type=bool, default=False)
    parser.add_argument('--show_progress', type=bool, default=False)
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()
    sequences_train, labels_train, characteristics_dict_train = load_data(args.amino_acids_file, args.train_data_file)
    sequences_test, labels_test, characteristics_dict_test = load_data(args.amino_acids_file, args.test_data_file)

    selected_features = args.features.split(",")

    X_train = [extract_features(seq, args.window_size, characteristics_dict_train, selected_features) for seq in sequences_train]
    X_test = [extract_features(seq, args.window_size, characteristics_dict_test, selected_features) for seq in sequences_test]
    
    if args.hyperparameter_tuning:
        best_c1, best_c2 = tune_hyperparameters(X_train, labels_train, X_test, labels_test, args.show_progress)
    else:
        best_c1, best_c2 = 0.1, 0.1

    crf = train_crf(X_train, labels_train, best_c1, best_c2)
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds.")
    accuracy = evaluate_crf(crf, X_test, labels_test)
    print(f"\nFinal Model -> c1={best_c1}, c2={best_c2}, Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(sklearn_crfsuite.metrics.flat_classification_report(labels_test, crf.predict(X_test), digits=4))

if __name__ == "__main__":
    main()