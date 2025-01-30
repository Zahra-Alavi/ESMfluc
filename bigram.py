from decimal import Decimal
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.linear_model import LogisticRegression
import argparse
import os
import pandas as pd

#################################################################
# DATA LOADING
#################################################################

data = pd.read_csv('neq_training_data.csv')

sequences = data['sequence']
neq_values = data['neq'].apply(lambda x: x.replace('[', '').replace(']', '').split(','))
neq_values = neq_values.apply(lambda x: [Decimal(i) for i in x])
amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # Standard amino acids
aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}  # Map amino acids to indices

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

#################################################################
# FEATURE ENGINEERING
#################################################################

# Extract features
def feature_extraction(sequences, neq_values):
    features = []
    targets = []
    for seq, neq_seq in zip(sequences, neq_values):
        for i, aa in enumerate(seq):
            # One-hot encoding
            aa_encoded = [0] * len(amino_acids)
            aa_encoded[aa_to_idx[aa]] = 1

            # Contextual features: Previous and next neq values (converted to float)
            prev_neq = float(neq_seq[i - 1]) if i > 0 else 0.0
            next_neq = float(neq_seq[i + 1]) if i < len(seq) - 1 else 0.0

            # Positional index (normalized)
            position = i / len(seq)

            features.append(aa_encoded + [prev_neq, next_neq, position])
            targets.append(float(neq_seq[i]))
    return features, targets
        
# Split the data into training, validation, and test sets    
X_train, X_test, y_train, y_test = train_test_split(sequences, neq_values, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

print(f"Number of data: {len(sequences)}")
print(f"Number of training samples: {len(X_train)}")
print(f"Number of validation samples: {len(X_val)}")
print(f"Number of test samples: {len(X_test)}")

X_train, y_train = feature_extraction(X_train, y_train)
X_val, y_val = feature_extraction(X_val, y_val)
X_test, y_test = feature_extraction(X_test, y_test)

#################################################################
# REGRESSION
#################################################################

def random_forest_regressor():
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

def logistic_regression_classifier():
    print("Training Logistic Regression Classifier...")
    y_train_class = classify_neq(y_train)
    y_val_class = classify_neq(y_val)
    y_test_class = classify_neq(y_test)
    
    model = LogisticRegression(random_state=42)
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
        "--classification",
        action="store_true",
        help="Train and evaluate a Logistic Regression Classifier."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tasks sequentially."
    )

    args = parser.parse_args()

    if args.all:
        print("Running all tasks...")
        data_learning()
        random_forest_regressor()
        logistic_regression_classifier()
    else:
        if args.data_learning:
            print("Running data learning...")
            data_learning()

        if args.regression:
            print("Running regression...")
            random_forest_regressor()

        if args.classification:
            print("Running classification...")
            logistic_regression_classifier()

    if not (args.all or args.data_learning or args.regression or args.classification):
        print("No task specified. Use --help for usage information.")

if __name__ == "__main__":
    main()
