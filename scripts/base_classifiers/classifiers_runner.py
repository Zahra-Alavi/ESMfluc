"""
Description: This script is used to run the classifiers on the dataset.
Date: 2025-02-07
Author: Ngoc Kim Ngan Tran
"""

import argparse
from data_loader import DataLoader
from classifiers_models import *
from data_learning import DataLearning

def main():
    parser = argparse.ArgumentParser(description="Run data analysis, feature engineering, regression, or classification tasks.")

    # Add arguments for tasks
    parser.add_argument(
        "--data_learning",
        action="store_true",
        help="Perform data learning, including histograms and amino acid analysis."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["RandomForestClassifier", "LogisticRegressionClassifier"],
        help="Choose the model to run."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tasks sequentially."
    )
    parser.add_argument(
        "--feature_engineering_version",
        type=str,
        default="1.1",
        help="Optional version number for feature engineering (default: 1.1)."
    )
    parser.add_argument(
        "--hyperameter_tuning",
        type=bool,
        default=False,
        help="Optional hyperparameter tuning for logistic regression/random forest model (default: False)."
    )

    args = parser.parse_args()
    if not (args.all or args.data_learning or args.model):
        print("No task specified. Use --help for usage information.")
    data_loader = DataLoader("../../data/neq_training_data.csv", args.feature_engineering_version, binary_classification=True)
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data()
    if args.all:
        print("Running all tasks...")
        DataLearning(data_loader.sequences, data_loader.neq_values).analyze_data()
        rf = RandomForestClassifierModel(X_train, y_train, X_test, y_test, args.hyperameter_tuning)
        print(rf.evaluate(rf.predict()))
        lr = LogisticRegressionClassifier(X_train, y_train, X_test, y_test, args.hyperameter_tuning)
        print(lr.evaluate(lr.predict()))
    else:
        if args.data_learning:
            print("Running data learning...")
            DataLearning(data_loader.sequences, data_loader.neq_values).analyze_data()
            print("Data learning complete.")

        if args.model == "RandomForestClassifier":
            print("Running Random Forest classifier...")
            rf = RandomForestClassifierModel(X_train, y_train, X_test, y_test, args.hyperameter_tuning)
            rf.fit()
            print(rf.evaluate(rf.predict()))
        elif args.model == "LogisticRegressionClassifier":
            print("Running Logistic Regression classifier...")
            lr = LogisticRegressionClassifier(X_train, y_train, X_test, y_test, args.hyperameter_tuning)
            lr.fit()
            print(lr.evaluate(lr.predict()))


if __name__ == "__main__":
    main()
