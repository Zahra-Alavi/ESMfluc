from decimal import Decimal
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.linear_model import LogisticRegression
import argparse
import os
import pandas as pd
import numpy as np
import esm
import torch

#################################################################
# DATA LOADING
#################################################################

data = pd.read_csv('neq_training_data.csv')

sequences = data['sequence']
neq_values = data['neq'].apply(lambda x: x.replace('[', '').replace(']', '').split(','))
neq_values = neq_values.apply(lambda x: [Decimal(i) for i in x])
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
#################################################################
# DATA LEARNING
#################################################################

def data_learning():
    if not os.path.exists("plot/data/neq"):
        os.makedirs("plot/data/neq")

    # Calculate sequence lengths
    lengths = [len(seq) for seq in sequences]
    plot_histogram(lengths, bins=50, title="Histogram of Sequence Lengths", xlabel="Sequence Length", ylabel="Frequency", save_as="plot/data/sequence_lengths.png")

    # Calculate the frequency of each amino acid
    aa_counts = {aa: 0 for aa in amino_acids}
    for seq in sequences:
        for aa in seq:
            aa_counts[aa] += 1
    print(aa_counts)
    plot_horizontal_bar(list(aa_counts.values()), list(aa_counts.keys()), title="Amino Acid Frequencies", xlabel="Frequency", ylabel="Amino Acids", save_as="plot/data/amino_acid_frequencies.png")

    # Calculate everage neq values for each amino acid
    aa_neq = {aa: [] for aa in amino_acids}
    for seq, neq_seq in zip(sequences, neq_values):
        for i, aa in enumerate(seq):
            aa_neq[aa].append(neq_seq[i])

    for aa in amino_acids:
        # plot probability distribution of neq values for each amino acid
        plot_histogram(aa_neq[aa], bins=50, title=f"Neq Distribution for {aa}", xlabel="Neq Value", ylabel="Frequency", save_as=f"plot/data/neq/neq_distribution_{aa}.png")
        
    # Plot average neq values for each amino acid
    aa_avg_neq = {aa: sum(neq_values) / len(neq_values) for aa, neq_values in aa_neq.items()}
    plot_horizontal_bar(list(aa_avg_neq.values()), list(aa_avg_neq.keys()), title="Average Neq Values for Amino Acids", xlabel="Average Neq Value", ylabel="Amino Acids", save_as="plot/data/neq/average_neq_values.png")
    
    ## Calculate amount of sequences start with 1.0 and 1.0 neq values
    print("neq values", neq_values)
    total_first_neq_with_1 = 0
    total_second_neq_with_1 = 0
    total_first_and_second_neq_with_1 = 0
    total_last_neq_with_1 = 0
    total_second_to_last_neq_with_1 = 0
    for neq_seq in neq_values:
        if neq_seq[0] == Decimal("1.0"):
            total_first_neq_with_1 += 1
        if neq_seq[1] == Decimal("1.0"):
            total_second_neq_with_1 += 1
        if neq_seq[0] == Decimal("1.0") and neq_seq[1] == Decimal("1.0"):
            total_first_and_second_neq_with_1 += 1
        if neq_seq[-1] == Decimal("1.0"):
            total_last_neq_with_1 += 1
        if neq_seq[-2] == Decimal("1.0"):
            total_second_to_last_neq_with_1 += 1
    
    print(f"Total sequences with 1.0 neq value at the start: {total_first_neq_with_1}")
    print(f"Total sequences with 1.0 neq value at the second position: {total_second_neq_with_1}")
    print(f"Total sequences with 1.0 neq value at the start and second position: {total_first_and_second_neq_with_1}")
    print(f"Total sequences with 1.0 neq value at the end: {total_last_neq_with_1}")
    print(f"Total sequences with 1.0 neq value at the second last position: {total_second_to_last_neq_with_1}")

#################################################################
# FEATURE ENGINEERING
#################################################################

def _one_hot_encode(aa):
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}  # Map amino acids to indices
    aa_encoded = [0] * len(amino_acids)
    aa_encoded[aa_to_idx[aa]] = 1
    return aa_encoded

# Extract features
def feature_extraction(sequences, neq_values, version="LR1.0"):
    features = []
    targets = []
    
    amino_acids_characteristics = pd.read_csv('amino_acids_characteristics.csv')
    amino_acids_characteristics.columns = amino_acids_characteristics.columns.str.strip()

    # Create a dictionary where the key is the amino acid and the value is a list of its characteristics.
    amino_acids_characteristics_dict = {row['Amino Acids']: [row['Charges'], row['Polar'], row['Hydrophobic']] for _, row in amino_acids_characteristics.iterrows()}
    sequences = sequences.reset_index(drop=True)

    if version == "1.3":
        # Load ESM model
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()

        # Move the model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Batch Processing
        batch_size = 16  # Adjust based on available memory
        seq_embedding = []

        for i in range(0, len(sequences), batch_size):
            batch_data = [(f"protein_{j}", sequences[j]) for j in range(i, min(i + batch_size, len(sequences)))]
            
            # Convert batch to tensor
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            
            batch_tokens = batch_tokens.to(device)

            # Extract per-residue representations
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[6])
                token_embedding = results["representations"][6]

            # Generate per-sequence representations via mean-pooling
            for j, (_, seq) in enumerate(batch_data):
                seq_len = len(seq)  # Actual sequence length (excluding special tokens)
                
                # Ignore [CLS] and [EOS] tokens
                seq_repr = token_embedding[j, 1:seq_len + 1].mean(dim=0)  
                seq_embedding.append(seq_repr)

        # Convert to numpy array
        seq_embedding = torch.stack(seq_embedding).cpu().numpy()
        features = seq_embedding
        targets = np.array(neq_values)

        print("Features shape:", features.shape)
        print("Targets shape:", targets.shape)
    else:
        for seq, neq_seq in zip(sequences, neq_values):
            for i, aa in enumerate(seq):
                # One-hot encoding
                aa_encoded = _one_hot_encode(aa)

                # Positional index (normalized)
                position = i / len(seq)
                targets.append(float(neq_seq[i]))
                
                if version == "LR1.0":
                    # Contextual features: Previous and next neq values (converted to float)
                    prev_neq = float(neq_seq[i - 1]) if i > 0 else 0.0
                    next_neq = float(neq_seq[i + 1]) if i < len(seq) - 1 else 0.0
                    features.append(aa_encoded + [prev_neq, next_neq, position])
                elif version == "LR1.1":
                    prev_aa = seq[i - 1] if i > 0 else -1
                    next_aa = seq[i + 1] if i < len(seq) - 1 else -1
                    
                    prev_aa_encoded = _one_hot_encode(prev_aa) if prev_aa != -1 else [0] * len(amino_acids)
                    next_aa_encoded = _one_hot_encode(next_aa) if next_aa != -1 else [0] * len(amino_acids)
                    features.append(aa_encoded + prev_aa_encoded +  next_aa_encoded + [position])
                elif version == "LR1.2":
                    prev_aa = seq[i - 1] if i > 0 else -1
                    next_aa = seq[i + 1] if i < len(seq) - 1 else -1
                    
                    prev_aa_encoded = _one_hot_encode(prev_aa) if prev_aa != -1 else [0] * len(amino_acids)
                    next_aa_encoded = _one_hot_encode(next_aa) if next_aa != -1 else [0] * len(amino_acids)
                    features.append(aa_encoded + prev_aa_encoded +  next_aa_encoded + [position] + amino_acids_characteristics_dict[aa])
                
    return features, targets
        
def load_data(version="LR1.0"):
    # Split the data into training, validation, and test sets    
    X_train, X_test, y_train, y_test = train_test_split(sequences, neq_values, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    print(f"Number of data: {len(sequences)}")
    print(f"Number of training samples: {len(X_train)}")
    print(f"Number of validation samples: {len(X_val)}")
    print(f"Number of test samples: {len(X_test)}")

    X_train, y_train = feature_extraction(X_train, y_train, version)
    X_val, y_val = feature_extraction(X_val, y_val, version)
    X_test, y_test = feature_extraction(X_test, y_test, version)
    return X_train, X_val, X_test, y_train, y_val, y_test

#################################################################
# REGRESSION
#################################################################

def random_forest_regressor(X_train, X_val, X_test, y_train, y_val, y_test):
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"Mean Squared Error: {mse}")


#################################################################
# CLASSIFICATION
#################################################################
# Split class into 2 classes: 0 and 1 - 0 for neq == 1.0 and 1 for neq > 1.0

def classify_neq(neq_values):
    return [0 if neq == Decimal("1.0") else 1 for neq in neq_values]

def logistic_regression_classifier(X_train, X_val, X_test, y_train, y_val, y_test, version="LR1.0"):
    print(f"Running classification with version {version}...")
    y_train_class = classify_neq(y_train)
    y_val_class = classify_neq(y_val)
    y_test_class = classify_neq(y_test)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train_class)

    y_pred = model.predict(X_val)
    print(classification_report(y_val_class, y_pred))
    
def random_forest_classifier(X_train, X_val, X_test, y_train, y_val, y_test):
    print("Training Random Forest Classifier...")
    y_train_class = classify_neq(y_train)
    y_val_class = classify_neq(y_val)
    y_test_class = classify_neq(y_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train_class)

    y_pred = model.predict(X_val)
    print(classification_report(y_val_class, y_pred))

#################################################################
# MAIN
#################################################################

def main():
    parser = argparse.ArgumentParser(description="Run data analysis, feature engineering, regression, or classification tasks.")

    # Add arguments for tasks
    parser.add_argument(
        "--data_learning",
        action="store_true",
        help="Perform data learning, including histograms and amino acid analysis."
    )
    parser.add_argument(
        "--regression",
        action="store_true",
        help="Train and evaluate a Random Forest Regressor."
    )
    parser.add_argument(
        "--classification_LR",
        action="store_true",
        help="Train and evaluate a Logistic Regression Classifier."
    )
    parser.add_argument(
        "--classification_RF",
        action="store_true",
        help="Train and evaluate a Random Forest Classifier."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tasks sequentially."
    )
    parser.add_argument(
        "--version",
        type=str,
        default="LR1.0",
        help="Optional version number (default: LR1.0)."
    )

    args = parser.parse_args()
    if not (args.all or args.data_learning or args.regression or args.classification_LR or args.classification_RF):
        print("No task specified. Use --help for usage information.")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(version=args.version)
    if args.all:
        print("Running all tasks...")
        data_learning()
        random_forest_regressor(X_train, X_val, X_test, y_train, y_val, y_test)
        logistic_regression_classifier(X_train, X_val, X_test, y_train, y_val, y_test, version=args.version)
    else:
        if args.data_learning:
            print("Running data learning...")
            data_learning()

        if args.regression:
            print("Running regression...")
            random_forest_regressor(X_train, X_val, X_test, y_train, y_val, y_test)

        if args.classification_LR:
            logistic_regression_classifier(X_train, X_val, X_test, y_train, y_val, y_test, version=args.version)
        
        if args.classification_RF:
            random_forest_classifier(X_train, X_val, X_test, y_train, y_val, y_test)

if __name__ == "__main__":
    main()
