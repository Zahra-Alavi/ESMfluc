"""
Description: This script is used to run the classifiers on the dataset.
Date: 2025-02-07
Author: Ngoc Kim Ngan Tran
"""

import argparse
import time
from data_loader import DataLoader
from classifiers_models import *
from data_learning import DataLearning
from esm_models_learning import ESMModelLearning
def main():
    parser = argparse.ArgumentParser(description="Run data analysis, feature engineering, regression, or classification tasks.")

    # Add arguments for tasks
    parser.add_argument(
        "--data_learning",
        action="store_true",
        help="Perform data learning, including histograms and amino acid analysis."
    )
    parser.add_argument(
        "--data_learning_file",
        type=str,
        default="../../data/neq_original_data.csv",
        help="Optional path to the data file for data learning (default: ../../data/neq_original_data.csv)."
    )
    parser.add_argument(
        "--train_data_file",
        type=str,
        default="../../data/train_data.csv",
        help="Optional path to the training data file (default: ../../data/train_data.csv)."
    )
    parser.add_argument(
        "--test_data_file",
        type=str,
        default="../../data/test_data.csv",
        help="Optional path to the test data file (default: ../../data/test_data.csv)."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["RandomForestClassifier", "LogisticRegressionClassifier"],
        help="Choose the model to run."
    )
    parser.add_argument(
        "--feature_engineering_version",
        type=str,
        default="1.1",
        help="Optional version number for feature engineering (default: 1.1)."
    )
    parser.add_argument(
        "--hyperameter_tuning",
        action="store_true",
        help="Optional hyperparameter tuning for logistic regression/random forest model (default: False)."
    )
    
    parser.add_argument(
        "--esm_model",
        type=str,
        default="esm2_t6_8M_UR50D",
        help="Optional ESM model to use for feature extraction (default: esm2_t6_8M_UR50D)."
    )
    
    parser.add_argument(
        "--esm_model_learning",
        action="store_true",
        help="Optional ESM model learning to find the best model for the task (default: False)."
    )

    args = parser.parse_args()
    if not (args.data_learning or args.model or args.esm_model_learning):
        print("No task specified. Use --help for usage information.")
    
    if args.data_learning:
        data_loader = DataLoader(args.data_learning_file, args.feature_engineering_version, args.esm_model, binary_classification=True)
        DataLearning(data_loader.sequences, data_loader.neq_values).analyze_data()

    if args.esm_model_learning:
        ESMModelLearning(args).run()
    
    if args.model:
        start_time = time.time()
        data_loader = DataLoader(args.train_data_file, args.feature_engineering_version, args.esm_model, binary_classification=True)
        X_train, y_train = data_loader.get_data()
        X_test, y_test = DataLoader(args.test_data_file, args.feature_engineering_version, args.esm_model, binary_classification=True).get_data()

        classifier = ClassifierFactory.get_classifier(args.model, X_train, y_train, X_test, y_test, args.hyperameter_tuning)
        classifier.fit()
        print(f"Training time: {time.time() - start_time:.2f} seconds")
        print(classifier.evaluate(classifier.predict()))
        
        if (args.model == "LogisticRegressionClassifier"):
            print(f"Model iterations: {classifier.model.n_iter_}")

if __name__ == "__main__":
    main()