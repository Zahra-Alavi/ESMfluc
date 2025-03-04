import csv
import os
from tabulate import tabulate
from data_loader import DataLoader
from classifiers_models import LogisticRegressionClassifier

class ESMModelLearning:
    def __init__(self, args):
        self.args = args
    
    def run(self):
        print("Running ESM model learning for different ESM models...")
        esm_models = [
            "esm1_t6_43M_UR50S", "esm1_t12_85M_UR50S", "esm1_t34_670M_UR100", "esm1_t34_670M_UR50D",
            "esm1_t34_670M_UR50S", "esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D",
            "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"
        ]
        
        # Ensure results directory exists
        results_dir = "../../results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Define the CSV file path
        csv_file = os.path.join(results_dir, "esm_model_results.csv")

        # Write the CSV header
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Class", "Precision", "Recall", "F1-Score", "Support"])

        # Store results for tabulate printing
        all_results = []

        for model in esm_models:
            X_train, y_train = DataLoader(self.args.train_data_file, self.args.feature_engineering_version, model, binary_classification=True).get_data() 
            X_test, y_test = DataLoader(self.args.test_data_file, self.args.feature_engineering_version, model, binary_classification=True).get_data()

            lr = LogisticRegressionClassifier(X_train, y_train, X_test, y_test, False)
            lr.fit()
            stats = lr.evaluate(lr.predict())

            # Collect results for each class
            results = [
                [model, "0", stats["0"]["precision"], stats["0"]["recall"], stats["0"]["f1-score"], stats["0"]["support"]],
                [model, "1", stats["1"]["precision"], stats["1"]["recall"], stats["1"]["f1-score"], stats["1"]["support"]]
            ]

            # Append results to the CSV file
            with open(csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(results)

            # Store results for tabulated printing
            all_results.extend(results)

        # Print results using tabulate
        print("\nESM Model Results:")
        print(tabulate(all_results, headers=["Model", "Class", "Precision", "Recall", "F1-Score", "Support"], tablefmt="github"))

        print(f"\nResults saved to {csv_file}")
