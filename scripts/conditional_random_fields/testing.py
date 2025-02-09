"""
Description:
Date: 2025-02-08
Author: Ngoc Kim Ngan Tran
"""

import pandas as pd
import argparse
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import sklearn_crfsuite
import ast
from decimal import Decimal

# Function to extract features from a sequence
def extract_features(sequence, window_size=2, characteristics_dict=None, selected_features=None):
    features = []
    seq_length = len(sequence)

    for i in range(seq_length):
        feature_dict = {
            "residue": sequence[i],
            "prev_residue": sequence[i-1] if i > 0 else "START",
            "next_residue": sequence[i+1] if i < seq_length - 1 else "END",
        }

        # Add selected characteristics based on user input
        if 'charges' in selected_features:
            feature_dict["charges"] = characteristics_dict[sequence[i]][0]
        if 'polar' in selected_features:
            feature_dict["polar"] = characteristics_dict[sequence[i]][1]
        if 'hydrophobic' in selected_features:
            feature_dict["hydrophobic"] = characteristics_dict[sequence[i]][2]
        if 'molecular_weight' in selected_features:
            feature_dict["molecular_weight"] = characteristics_dict[sequence[i]][3]
        if 'pKa' in selected_features:
            feature_dict["pKa"] = characteristics_dict[sequence[i]][7]
        if 'pKb' in selected_features:
            feature_dict["pKb"] = characteristics_dict[sequence[i]][8]
        if 'pKx' in selected_features:
            feature_dict["pKx"] = characteristics_dict[sequence[i]][9]
        if 'pI' in selected_features:
            feature_dict["pI"] = characteristics_dict[sequence[i]][10]

        # Add a sliding window of residues
        for w in range(1, window_size + 1):
            feature_dict[f"prev_{w}_residue"] = sequence[i - w] if i - w >= 0 else "START"
            feature_dict[f"next_{w}_residue"] = sequence[i + w] if i + w < seq_length else "END"

        features.append(feature_dict)

    return features

# Function to classify Neq values
def classify_neq(neq_values):
    return ["0" if Decimal(neq) == Decimal("1.0") else "1" for neq in neq_values]

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a CRF model for predicting Neq values of amino acids.")
    parser.add_argument('--amino_acids_file', type=str, default="../../data/amino_acids_characteristics.csv", help='Path to the amino acids characteristics CSV file.')
    parser.add_argument('--training_data_file', type=str, default="../../data/neq_training_data.csv", help='Path to the training data CSV file.')
    parser.add_argument('--window_size', type=int, default=2, help='Window size for feature extraction (default is 2).')
    parser.add_argument('--features', type=str, default="charges,polar,hydrophobic", help='Comma-separated list of features to use (choices: charges, polar, hydrophobic, molecular_weight, pKa, pKb, pKx, pI).')
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()

    # Read in amino acid characteristics
    amino_acids_characteristics = pd.read_csv(args.amino_acids_file)

    # Build a dictionary for amino acid characteristics
    characteristics_dict = {
        row['Amino Acids']: [row['Charges'], row['Polar'], row['Hydrophobic'], row['Molecular Weight'], row['Molecular Formula'], row['Residue Formula'], row['Residue Weight'], row['pKa'], row['pKb'], row['pKx'], row['pI']]
        for _, row in amino_acids_characteristics.iterrows()
    }

    # Load training data
    data = pd.read_csv(args.training_data_file)
    sequences = data["sequence"].tolist()

    # Convert the string representation of lists to actual lists of floats
    labels = data['neq'].apply(lambda x: list(map(float, ast.literal_eval(x))))
    labels = labels.apply(classify_neq)

    # Extract features for each sequence
    selected_features = args.features.split(",")  # List of features user wants to use
    X = [extract_features(seq, window_size=args.window_size, characteristics_dict=characteristics_dict, selected_features=selected_features) for seq in sequences]

    # Create labels
    y = labels.tolist()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train CRF model
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(X_train, y_train)
    y_pred = crf.predict(X_test)

    # Flatten y_test and y_pred (i.e., turn the list of lists into a single list)
    y_test_flat = [item for sublist in y_test for item in sublist]
    y_pred_flat = [item for sublist in y_pred for item in sublist]

    # Calculate performance metrics
    print("Classification Report:")
    print(classification_report(y_test_flat, y_pred_flat))

if __name__ == "__main__":
    main()




