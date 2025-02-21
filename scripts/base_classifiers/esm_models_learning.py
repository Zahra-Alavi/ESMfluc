from data_loader import DataLoader
from classifiers_models import LogisticRegressionClassifier
import os
from tabulate import tabulate

class ESMModelLearning:
    def __init__(self):
        pass
    
    def run(self):
        print("Running ESM model learning for different ESM models...")
        # esm_models = ["esm1_t6_43M_UR50S", "esm1_t12_85M_UR50S", "esm1_t34_670M_UR100", "esm1_t34_670M_UR50D", "esm1_t34_670M_UR50S","esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"]
        esm_models = ["esm2_t48_15B_UR50D"]
        results = []
        if not os.path.exists("../../results"):
            os.makedirs("../../results")
        
        # write the header of the file
        with open("../../results/esm_model_results.txt", "w") as f:
            f.write("ESM Model Results\n")
            f.write(tabulate([], headers=["Model", "Class", "Accuracy", "Macro F1", "Weighted F1"], tablefmt="github"))
        for model in esm_models:
            data_loader = DataLoader("../../data/neq_training_data.csv", "1.3", model, binary_classification=True)
            X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data()
            lr = LogisticRegressionClassifier(X_train, y_train, X_test, y_test, False)
            lr.fit()
            stats = lr.evaluate(lr.predict())
            results = []
            results.append([model, "0", stats["0"]["precision"], stats["0"]["recall"], stats["0"]["f1-score"], stats["0"]["support"]])
            results.append(["-", "-", "-", "-", "-"])
            results.append(["", "1", stats["1"]["precision"], stats["1"]["recall"], stats["1"]["f1-score"], stats["1"]["support"]])

            with open("../../results/esm_model_results.txt", "a") as f:
                f.write("\n")   
                f.write(tabulate(results, tablefmt="github"))
