from data_loader import DataLoader
from classifiers_models import LogisticRegressionClassifier
import os
from tabulate import tabulate

class ESMModelLearning:
    def __init__(self):
        pass
    
    def run(self):
        print("Running ESM model learning for different ESM models...")
        esm_models = ["esm1_t6_43M_UR50S", "esm1_t12_85M_UR50S", "esm1_t34_670M_UR100", "esm1_t34_670M_UR50D", "esm1_t34_670M_UR50S", "esm1b_t33_650M_UR50S", "esm_msa1_t12_100M_UR50S", "esm_msa1b_t12_100M_UR50S", "esm1v_t33_650M_UR90S_[1-5]", "esm_if1_gvp4_t16_142M_UR50", "esmfold_v0", "esmfold_v1", "esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D", "	esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"]
        results = []
        for model in esm_models:
            data_loader = DataLoader("../../data/neq_training_data.csv", "1.3", model, binary_classification=True)
            X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data()
            lr = LogisticRegressionClassifier(X_train, y_train, X_test, y_test, False)
            stats = lr.evaluate(lr.predict())
            results.append([model, stats['accuracy'], stats['macro avg']['f1-score'], stats['weighted avg']['f1-score']])
        
        # Save results to a file
        if not os.path.exists("../../results"):
            os.makedirs("../../results")
        with open("../../results/esm_model_results.txt", "w") as f:
            f.write(tabulate(results, headers=["Model", "Accuracy", "Macro F1", "Weighted F1"], tablefmt="github"))
    
        
    